from typing import Dict, Iterable, List, Optional
from pathlib import Path

from discoket.ccg2discocat.ccg_parser import CCGParser
from discoket.ccg2discocat.ccg_tree import CCGTree
from discoket.ccg2discocat.ccgbank.ccgbank_utils import *

from discopy import Diagram


class CCGBankParser(CCGParser):
    """A parser for CCGBank trees.

    Note that generalized composition rules (present in CCGBank)
    cannot be handled by the existing infrastructure. These rules
    appear in the resulting tree as CCGRule.UNKNOWN, which means
    conversion to DisCoCat diagrams is not possible for these
    cases.
    """

    def __init__(self, path_to_auto_trees: str):
        """Instantiate a CCGBank parser.

        Parameters
        ----------
        path_to_auto_trees : str
            The path to the directory for the AUTO trees. This is usually:
                ccgbank_X_X/data/AUTO
        """
        self.path = Path(path_to_auto_trees)

    def _extract_root(self, tree: List[str]) -> CCGBankNode:
        assert tree[0].startswith('(<')
        node_type = tree[0][2]
        if node_type == CCGBankNodeType.TREE:
            node = CCGBankNode(node_type, tree[1], "", int(tree[2]), int(tree[3][:-1]))
        else:
            node = CCGBankNode(node_type, tree[1], tree[4], 0, 0)
        return node

    def _extract_children(self, tree: List[str]) -> List[List[str]]:
        par_count = 0
        end_indices = []
        for idx, tok in enumerate(tree[4:]):
            if tok.startswith('(<'):
                par_count += 1
            elif tok.endswith('>)') or tok == ")":
                par_count -= 1
                if par_count == 0:
                    end_indices.append(idx+4)

        return [tree[4: end_indices[0]+1]] + [
            tree[end_indices[i]+1:end_indices[i+1]+1]
            for i in range(0, len(end_indices)-1)
        ]

    def _get_ccgtree(self, tree: List[str]) -> CCGTree:
        root = self._extract_root(tree)
        if root.type == CCGBankNodeType.TREE:
            children = self._extract_children(tree)
            ccgtree = CCGTree(
                text=None,
                rule=CCGRule.UNKNOWN,
                biclosed_type=ccg_to_biclosed(root.cat),
                children=list((map(self._get_ccgtree, children)))
            )
            rule = determine_ccg_rule(ccgtree.biclosed_type, [ch.biclosed_type for ch in ccgtree.children])
            ccgtree.rule = rule
        else:
            ccgtree = CCGTree(
                text=root.token,
                rule=CCGRule.LEXICAL,
                biclosed_type=ccg_to_biclosed(root.cat),
                children=[]
            )
        return ccgtree

    def build_ccgtree(self, ccgbank_tree: str) -> CCGTree:
        """Builds a CCGTree from an AUTO tree in string form."""
        tree = ccgbank_tree.split()
        return self._get_ccgtree(tree)

    def to_diagram(self, ccgbank_tree: str) -> Optional[Diagram]:
        """Creates a DisCoPy diagram from an AUTO tree in string form."""
        ccg_tree = self.build_ccgtree(ccgbank_tree)
        return ccg_tree.to_diagram()

    def parse_section(self, section_id: int, suppress_exceptions: bool = False) -> Dict[str, Optional[Diagram]]:
        """Parses a CCGBank section."""
        auto_trees = {
            entry[0]: entry[1]
            for entry in read_ccgbank_section(self.path, section_id)
        }
        ccg_trees = {
            id: self.build_ccgtree(auto_trees[id])
            for id in auto_trees.keys()
        }

        diagrams = {}
        for tree_id in ccg_trees:
            try:
                diagrams[tree_id] = ccg_trees[tree_id].to_diagram()
            except Exception as er:
                if suppress_exceptions:
                    diagrams[tree_id] = None
                else:
                    raise er
        return diagrams

    def sentences2trees(self,
                        sentences: Iterable[str],
                        suppress_exceptions: bool = False) -> List[Optional[CCGTree]]:
        raise NotImplementedError("Please use `build_ccgtree` or `parse_section`.")

    def sentences2diagrams(
            self,
            sentences: Iterable[str],
            planar: bool = False,
            suppress_exceptions: bool = False) -> List[Optional[Diagram]]:
        raise NotImplementedError("Please use `to_diagram` or `parse_section`.")


if __name__ == "__main__":

    parser = CCGBankParser("/home/dimkart/corpora/ccgbank_1_1/data/AUTO")
    trees = parser.parse_section(0, suppress_exceptions=True)

    print()

