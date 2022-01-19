# Copyright 2021 Cambridge Quantum Computing Ltd.
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

from typing import List, Tuple, Optional

from discopy import Cup, Diagram, Id, Swap, Ty, Word


def is_pregroup_diagram(diagram: Diagram) -> bool:
    """Check if a DisCoPy diagram is a pregroup diagram. This
    code is taken from `discopy.grammar.pregroup.draw()`.

    Parameters
    ----------
    diagram: discopy.Diagram
        The diagram to be checked.

    Returns
    -------
    bool
        True if the diagram is a pregroup diagram, and False
        otherwise.
    """
    words, is_pregroup = Id(Ty()), True
    for _, box, right in diagram.layers:
        if isinstance(box, Word):
            if right:  # word boxes should be tensored left to right.
                is_pregroup = False
                break
            words = words @ box
        else:
            break
    cups = diagram[len(words):].foliation().boxes\
        if len(words) < len(diagram) else []
    is_pregroup = is_pregroup and words and all(
        isinstance(box, (Cup, Swap))
        for s in cups for box in s.boxes)

    return is_pregroup


def create_pregroup_diagram(
    words: List[Word],
    cod: Ty,
    morphisms: Optional[List[Tuple[type, int, int]]] = None
) -> Diagram:
    """Create a DisCoPy pregroup diagram from a list of cups and swaps.

    Parameters
    ----------
    words: A list of :py:class:`discopy.Word` \s.
        A list of :py:class:`discopy.Word` objects corresponding to
        the words of the sentence.
    cod: discopy.Ty
        The output type of the diagram.
    morphisms: A list of `Tuple[type, int, int]` or None, default = None
        A list of tuples of the form (morphism, start_wire_idx, end_wire_idx).
        Morphisms can be from :py:class:`discopy.Cup` and
        :py:class:`discopy.Swap`, while the two numbers define the indices of
        the wires on which the morphism is applied. The index range is [1..n],
        where 1 correspond to the first atomic type of the first word, and `n`
        to the last atomic type of the last word.

    Returns
    -------
    :py:obj:`discopy.Diagram`
        The generated pregroup diagram.

    Raises
    ------
    :py:class:`discopy.cat.AxiomError`
        If the provided morphism list does not type-check properly.
    """
    if morphisms is None:
        morphisms = []

    types: Ty = Ty()
    boxes: List[Word] = []
    offsets: List[int] = []
    for w in words:
        boxes.append(w)
        offsets.append(len(types))
        types @= w.cod

    for idx, (typ, start, end) in enumerate(morphisms):
        if typ not in (Cup, Swap):
            raise ValueError(f"Unknown morphism type: {typ}")
        box = typ(types[start:start+1], types[end:end+1])

        boxes.append(box)
        actual_idx = start
        for pr_idx in range(idx):
            if morphisms[pr_idx][0] == Cup and \
                    morphisms[pr_idx][1] < start:
                actual_idx -= 2
        offsets.append(actual_idx)

    return Diagram(dom=Ty(), cod=cod, boxes=boxes, offsets=offsets)
