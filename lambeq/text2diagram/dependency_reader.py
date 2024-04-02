# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = ['DependencyReader']

from collections.abc import Callable
from dataclasses import dataclass, field
import re

from lambeq.backend.grammar import Box, Diagram, Ty
from lambeq.core.utils import SentenceType
from lambeq.text2diagram.base import Reader
from lambeq.text2diagram.bobcat_parser import BobcatParser
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree


@dataclass(order=True)
class SignedDigits:
    """A representation of numbers used for ordering.

    Each place digit is infinitesimally smaller than the previous digit,
    and flips each time the direction of counting changes. Therefore,
    (m, -n) is always between (m-1,) and (m,) no matter how large n is.
    """
    digits: tuple[int, ...] = (0,)  # Must end in 0 for comparison

    def __add__(self, other: int) -> SignedDigits:
        if not other:
            return self

        digits = self.digits
        if len(digits) % 2:  # counting down
            return SignedDigits((*digits[:-1], other, 0))
        else:  # counting up
            return SignedDigits((*digits[:-2], digits[-2] + other, 0))

    def __sub__(self, other: int) -> SignedDigits:
        if not other:
            return self

        digits = self.digits
        if len(digits) % 2:  # counting down
            return SignedDigits((*digits[:-2], digits[-2] - other, 0))
        else:  # counting up
            return SignedDigits((*digits[:-1], -other, 0))


@dataclass
class Dependency:
    """A dependency in the dependency graph."""
    parent: int
    child: int
    relation: str

    def ty(self, indices: bool = True, relation: bool = False) -> Ty:
        output = ''
        if indices:
            output += f'{self.parent} -> {self.child}'
            if relation:
                output += ': '
        if relation:
            output += self.relation
        return Ty(output)


@dataclass
class DepNode:
    """A node in the dependency graph.

    The node initially just contains the word and its index; the rest of
    the data is filled in as the graph is built.
    """
    index: int
    word: str

    # An unordered list of the parent and children nodes
    parents: list[DepNode] = field(default_factory=list)
    children: list[DepNode] = field(default_factory=list)

    # Position of the node in the graph, once topologically sorted
    position: int | None = None

    # Transitive closure, i.e. all reachable children
    closure: set[int] = field(default_factory=set)

    # Approximate height of the node, calculated by traversing from root
    height: SignedDigits | None = None

    # (Approximate) Height of the farthest away child
    max_child: SignedDigits | None = None

    # The order of the indices of the parents and the children.
    # Initially empty, this is filled in as the graph is built.
    parent_order: list[int] = field(default_factory=list)
    child_order: list[int] = field(default_factory=list)

    # The number of swaps required to connect this node to each child.
    swaps: list[int] = field(default_factory=list)

    # A running count of the number of wires covering this node on the
    # left and the right as the graph is built, then used to populate
    # the `swaps` list.
    left_wires: int = 0
    right_wires: int = 0

    def __str__(self) -> str:
        return f'{self.index}: {self.word}'

    @property
    def dangling(self) -> bool:
        # if there are any children that haven't been connected
        return len(self.children) > len(self.child_order)


class DependencyReader(Reader):
    r"""A reader that displays the dependency graph of a sentence.

    Dependencies represent a relation from a word that has a complex CCG
    type, and a word that fulfils an argument of that type.
    For example, in the sentence "John resigned", the verb "resigned"
    has type `S\NP` with one argument. This argument is fulfilled by the
    noun "John" which has type `NP`. Therefore, this sentence contains
    one dependency relation from "resigned" to "John".

    Each sentence has a list of dependencies. For example, the one for
    "John resigned" is, represented in notation:
    [ ⟨ ⟨ S\NP, "resigned" ⟩ , 1 , ⟨ NP "John" ⟩ ⟩ ]

    This reader converts this list into a diagram, where each word is a
    box, and each dependency is a labelled wire from the output of the
    argument word-box to the input of the predicate word-box.

    The dependencies are provided by a CCG parser; for example, Bobcat
    produces trees that contain dependency data, trained from the
    CCGBank corpus.

    """

    def __init__(
        self,
        ccg_parser: CCGParser | Callable[[], CCGParser] = BobcatParser,
        label_relations: bool = True,
        label_indices: bool = False,
        label_box_indices: bool = False,
        head_dependency: str = 'HEAD'
    ) -> None:
        """Initialise a dependency reader.

        Parameters
        ----------
        ccg_parser : CCGParser or callable, default: BobcatParser
            A :py:class:`CCGParser` object or a function that returns
            it. The parse tree produced by the parser is used to
            generate the tree diagram.
        label_relations : bool, default: True
            Whether to label the dependencies in the graph with the name
            of the relation.
        label_indices : bool, default: False
            Whether to label the dependencies in the graph with the
            indices of the head word and the filler word.
        label_box_indices : bool, default: False
            Whether to label the words in the graph with the index of
            their position in the sentence. Note: some indices may be
            missing because that word does not have any dependency
            relationships.
        head_dependency : str, default: 'HEAD'
            The name of the head dependency, used for the type of the
            output wire from the root word of the diagram.

        """
        if not isinstance(ccg_parser, CCGParser):
            if not callable(ccg_parser):
                raise ValueError(f'{ccg_parser} should be a CCGParser or a '
                                 'function that returns a CCGParser.')
            ccg_parser = ccg_parser()
            if not isinstance(ccg_parser, CCGParser):
                raise ValueError(f'{ccg_parser} should be a CCGParser or a '
                                 'function that returns a CCGParser.')

        self.ccg_parser = ccg_parser
        self.label_relations = label_relations
        self.label_indices = label_indices
        self.label_box_indices = label_box_indices
        self.head_dependency = head_dependency

    @staticmethod
    def tree2diagram(tree: CCGTree,
                     label_relations: bool,
                     label_indices: bool,
                     label_box_indices: bool,
                     head_dependency: str,
                     suppress_exceptions: bool = False) -> Diagram | None:
        """Convert a :py:class:`~.CCGTree` into a
        :py:class:`~lambeq.backend.grammar.Diagram` .

        The diagram shows the dependency graph of the parsed sentence.

        Core implementation in `DependencyReader._tree2diagram`.

        Parameters
        ----------
        tree : :py:class:`~.CCGTree`
            The CCG tree to be converted.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.

        Other Parameters
        ----------------
        See description for `DependencyReader.__init__`.

        Returns
        -------
        :py:class:`lambeq.backend.grammar.Diagram` or None
            The parsed diagram, or :py:obj:`None` on failure.

        """

        try:
            return DependencyReader._tree2diagram(tree._resolved(),
                                                  label_relations,
                                                  label_indices,
                                                  label_box_indices,
                                                  head_dependency)
        except Exception as e:
            if suppress_exceptions:
                return None
            else:
                raise e

    @staticmethod
    def _tree2diagram(tree: CCGTree,
                      label_relations: bool,
                      label_indices: bool,
                      label_box_indices: bool,
                      head_dependency: str) -> Diagram:
        """Helper function for `DependencyReader.tree2diagram`."""

        # Turn list of dependencies to list of nodes
        dependencies = {}
        nodes = [DepNode(i, word) for i, word in enumerate(tree.text.split())]
        for dep in tree.metadata['original'].deps:
            # Bobcat 1-indexes instead of 0-indexing
            parent = dep.head.index - 1
            child = dep.filler.index - 1
            assert (parent, child) not in dependencies

            # Clean the relation category name:
            # - remove variable data, e.g. "N{_}" -> "N"
            # - remove slot data, e.g. "S<1>" -> "S", except the one
            #   fulfilled by this dependency
            regex_pattern = fr'(\{{.\}})|(<(?!{dep.relation.slot}).>)'
            relation = re.sub(regex_pattern, '', dep.relation.category)

            dependency = Dependency(parent, child, relation)
            dependencies[parent, child] = dependency

            nodes[child].parents.append(nodes[parent])
            nodes[parent].children.append(nodes[child])

        # Get root index
        r = tree.root().metadata['original'].index - 1
        root = nodes[r]

        # Add head dependency to root as a parent
        assert not root.parents
        dependencies[r, r] = Dependency(r, r, head_dependency)
        root.parent_order.append(r)

        # Calculate approximate height of each node
        queue = [(root, None, SignedDigits())]
        while queue:
            node, child, height = queue.pop(0)
            is_parent = child is not None

            if is_parent:
                if node.index in child.closure:
                    raise NotImplementedError('Dependency cycle detected in: '
                                              f'{tree.text}')
                node.closure.update({child.index}, child.closure)

            # IF node height has not already been calculated
            # OR node height has been calculated, but is inconsistent
            #    with the previous node (e.g. not below its parent)
            # THEN calculate new height and add connected nodes to queue
            if node.height is None or not (
                node.height < height if is_parent else node.height > height
            ):
                node.height = height = height - 1 if is_parent else height + 1
                queue.extend((parent, node, height) for parent in node.parents)
                queue.extend((child, None, height) for child in node.children)

        # At this point, height comparisons are consistent
        # but make sure root always has the lowest height
        root.height = min(filter(None, (node.height for node in nodes))) - 1

        # For each node, calculate its approximate farthest child
        for node in nodes:
            if node.children:
                node.max_child = max((child.height, -child.index)
                                     for child in node.children)

        # Topologically sort the nodes using a stack.
        # Starting from the root node, consider any missing parents,
        # then insert the node and then consider its children.
        # This helps reduce the distance between the nodes.
        # The parents are considered starting with the one with the
        # highest child. The children are considered starting with the
        # one with the highest parent.
        added = set()
        node_order: list[DepNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.index in added:
                continue

            missing_parents = [parent for parent in node.parents
                               if parent.index not in added]
            if missing_parents:
                # Add parents to stack starting with the one with the
                # (approximately) closest child
                stack.extend(sorted(missing_parents,
                                    key=lambda parent: (parent.max_child,
                                                        parent.index)))
            else:
                # Add node to graph
                node.position = len(node_order)
                node_order.append(node)
                added.add(node.index)

                # Add children to stack, starting with the one with the
                # (approximately or actual) closest parent
                to_add = (c for c in node.children if c.index not in added)
                stack.extend(sorted(to_add, key=lambda child: (
                    min(p.position for p in child.parents
                        if p.position is not None),
                    child.index
                )))

        # Order nodes in a straight line and connect them either on the
        # left or right using as few swaps as possible
        for node in node_order:
            # We connect to each parent starting from the nearest
            parents = sorted(node.parents,
                             key=lambda p: p.position,
                             reverse=True)

            # For each parent, decide whether to connect on the left or
            # right. `fallback` is used to break ties.
            fallback = sum(p.right_wires - p.left_wires for p in parents) > 0
            for parent in parents:
                # The nodes in between this parent and the current node
                in_nodes = node_order[parent.position + 1:node.position]

                if not (any(in_node.dangling for in_node in in_nodes)
                        or parent.left_wires and parent.right_wires):
                    # The in-between nodes have no dangling edges
                    # so we are free to cover them up however we want
                    left = bool(parent.right_wires or (
                        not parent.left_wires
                        and (parent.index < node.parent_order[0]
                             if node.parent_order
                             else node.index < sum(parent.child_order[:1]))
                    ))
                elif parent.left_wires != parent.right_wires:
                    # Covering up a node adds one more swap
                    # (per dangling edge), so minimise if possible
                    left = parent.left_wires < parent.right_wires
                else:
                    # If the same number of swaps would be added, choose
                    # the side where the most nodes are already covered.
                    covers_left = sum(
                        (not in_node.left_wires) - (not in_node.right_wires)
                        for in_node in in_nodes if in_node.dangling
                    )
                    left = covers_left < 0 or covers_left == 0 and fallback

                # Now connect the parent
                if left:
                    node.parent_order.insert(0, parent.index)
                    parent.child_order.insert(0, node.index)
                    parent.swaps.insert(0, -parent.left_wires)
                    for in_node in in_nodes:
                        in_node.left_wires += 1
                else:
                    node.parent_order.append(parent.index)
                    parent.child_order.append(node.index)
                    parent.swaps.append(parent.right_wires)
                    for in_node in in_nodes:
                        in_node.right_wires += 1

        # Build diagram (domain is children, and codomain is parents)
        diagram = Diagram.id()
        for node in reversed(node_order):
            start = node.left_wires

            # Add swaps
            if any(node.swaps):
                permutation = [*range(len(diagram.cod))]
                permutation[start:start] = [permutation.pop(start + swap)
                                            for swap in node.swaps]
                diagram = diagram.permuted(permutation)

            # Add box
            name = str(node) if label_box_indices else node.word
            dom = Ty().tensor(dependencies[node.index, child].ty(
                label_indices, label_relations
            ) for child in node.child_order)
            cod = Ty().tensor(dependencies[parent, node.index].ty(
                label_indices, label_relations
            ) for parent in node.parent_order)
            box = Box(name, dom, cod)
            diagram = diagram.then_at(box, start)
        return diagram

    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False,
                         suppress_exceptions: bool = False) -> Diagram | None:
        """Parse a sentence into a lambeq diagram.

        The diagram shows the dependency graph of the parsed sentence.

        Parameters
        ----------
        sentence : str or list of str
            The sentence to be parsed.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.

        Returns
        -------
        :py:class:`lambeq.backend.grammar.Diagram` or None
            The parsed diagram, or :py:obj:`None` on failure.

        """
        tree = self.ccg_parser.sentence2tree(
            sentence=sentence,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions
        )

        if tree is None:
            return None
        return self.tree2diagram(
            tree,
            label_relations=self.label_relations,
            label_indices=self.label_indices,
            label_box_indices=self.label_box_indices,
            head_dependency=self.head_dependency,
            suppress_exceptions=suppress_exceptions
        )
