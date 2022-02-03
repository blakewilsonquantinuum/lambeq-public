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

"""
Text printer
------------

Module that allows printing of DisCoPy pregroup diagrams in text form,
e.g. for the purpose of outputting them graphically in a terminal.

"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

from discopy import Box, Cup, Diagram, Swap
from discopy.grammar import Word
from lambeq.pregroups import is_pregroup_diagram


def diagram2str(diagram: Diagram, word_spacing: int = 2,
                discopy_types: bool = False, compress_layers: bool = True,
                use_ascii: bool = False) -> str:
    """Produces a string that graphically represents the input diagram
    with text characters, without the need of first creating a printer.
    For specific arguments, see the constructor of the
    :py:class:`.TextDiagramPrinter` class."""
    printer = TextDiagramPrinter(word_spacing, discopy_types,
                                 compress_layers, use_ascii)
    return printer.diagram2str(diagram)


class _MorphismType(Enum):
    """Enumeration for expected morphism types in a diagram."""
    CUP = 1
    SWAP = 2
    SWAP_BOTTOM = 3  # Helper type for adding a 2nd printing row to swaps


@dataclass
class _Morphism:
    """Represents a morphism. `start` and `end` refer to the original
    positions of the involved atomic types in the diagram."""
    morphism: _MorphismType
    start: int
    end: int


class _Layer:
    """Represents a layer as a collection of `Morphism`s, and provides
    helper methods for the printing."""
    def __init__(self, morphisms: list[_Morphism]):
        self.morphisms = morphisms

    def morphism_allowed(self, morphism: _Morphism):
        """Checks if there is an overlap between the input morphism
        and the morphisms at the current layer."""
        return all([len(set(range(morphism.start, morphism.end + 1)) &
                    set(range(m.start, m.end + 1))) == 0
                    for m in self.morphisms])

    def sort_morphisms(self):
        self.morphisms = sorted(self.morphisms, key=lambda m: m.start)

    def is_empty(self):
        return len(self.morphisms) == 0


class TextDiagramPrinter:
    """A text printer for pregroup diagrams."""

    UNICODE_CHAR_SET: dict[str, str] = {
        "BAR": '│',
        "TOP_R_CORNER": '╮',
        "TOP_L_CORNER": '╭',
        "BOTTOM_L_CORNER": '╰',
        "BOTTOM_R_CORNER": '╯',
        "LINE": '─',
        "DOT": '·'
    }

    ASCII_CHAR_SET: dict[str, str] = {
        "BAR": '|',
        "TOP_R_CORNER": chr(160),
        "TOP_L_CORNER": chr(160),
        "BOTTOM_L_CORNER": '\\',
        "BOTTOM_R_CORNER": '/',
        "LINE": '_',
        "DOT": ' '
    }

    def __init__(self, word_spacing: int = 2, discopy_types: bool = False,
                 compress_layers: bool = True, use_ascii: bool = False):
        """Initialise a text diagram printer.

        Parameters
        ----------
        word_spacing : int
            The number of spaces between the words of the diagrams
            (default = 2).
        discopy_types : bool
            If true, types are represented in DisCoPy form (using @ as the
            monoidal product). Default is false, which present types in a
            more compact form.
        compress_layers : bool
            If true (the default value), it tries to removes the empty space
            from the graphical representation of the diagram providing a more
            compact output. If set to false, layers in the diagram are
            presented as found, i.e. a single cup/swap per layer.
        use_ascii: bool
            If true, drawing takes place with simple ASCII characters (e.g.
            to avoid problems in certain terminals). If false (default), Unicode
            characters are used for smoother printing.
        """
        self.word_spacing = word_spacing
        self.discopy_types = discopy_types
        self.compress_layers = compress_layers
        self.chr_set = self.UNICODE_CHAR_SET if not use_ascii \
            else self.ASCII_CHAR_SET

    @staticmethod
    def _induce_layers(atomic_types: list[str],
                       offsets: list[tuple[Box, int]]) -> list[_Layer]:
        """Prepares a list of `_Layer`s from the DisCoPy diagram, in which
        the start and end of each cup/swap point to the original positions
        of the involved atomic types (before any application of cups)."""
        typs = [(t, idx) for idx, t in enumerate(atomic_types)]
        layers = []
        for (box, ofs) in offsets:
            if isinstance(box, Cup):
                layers.append(_Layer([_Morphism(_MorphismType.CUP,
                                                typs[ofs][1],
                                                typs[ofs + 1][1])]))
                del typs[ofs:ofs + 2]
            else:  # isinstance(b, Swap):
                # Swaps are rendered by 2 layers
                layers += [
                    _Layer([_Morphism(_MorphismType.SWAP, typs[ofs][1],
                                      typs[ofs + 1][1])]),
                    _Layer([_Morphism(_MorphismType.SWAP_BOTTOM, typs[ofs][1],
                                      typs[ofs + 1][1])]),
                ]
        return layers

    @staticmethod
    def _compress_layers(layers: list[_Layer]) -> list[_Layer]:
        """Compresses the layers of the diagram from top to bottom."""
        new_layers = deepcopy(layers)
        for l_idx in range(1, len(layers)):
            if new_layers[l_idx].is_empty():
                continue
            m = new_layers[l_idx].morphisms[0]
            prev_idx = l_idx - 1
            while prev_idx >= 0 and new_layers[prev_idx].morphism_allowed(m):
                new_layers[prev_idx].morphisms.append(m)
                if m.morphism == _MorphismType.SWAP:
                    new_layers[prev_idx + 1].morphisms = \
                        new_layers[prev_idx + 1].morphisms[:-1] + \
                        [new_layers[prev_idx + 2].morphisms[-1]]
                    new_layers[prev_idx + 2].morphisms = \
                        new_layers[prev_idx + 2].morphisms[:-1]
                else:  # m.morphism == _MorphismType.Cup
                    new_layers[prev_idx + 1].morphisms = \
                        new_layers[prev_idx + 1].morphisms[:-1]
                prev_idx -= 1

        for l in new_layers:
            l.sort_morphisms()

        return [l for l in new_layers if not l.is_empty()]

    def _add_identities(self, previous_rows: list[str], current_row: str,
                        type_positions: list[int], start: int, end: int) -> str:
        """Scans the current printing row from left to right in the given part
        and adds identities when nececessary, based on the previous row."""
        for typ_id in range(start, end):
            pos = type_positions[typ_id]
            if (len(previous_rows) == 0 or previous_rows[-1][pos] in
                    [self.chr_set["BAR"], self.chr_set["TOP_L_CORNER"],
                     self.chr_set["TOP_R_CORNER"]]):
                current_row = self._replace_substr(current_row,
                                                   self.chr_set["BAR"],
                                                   pos, pos + 1)
        return current_row

    @staticmethod
    def _replace_substr(string: str, substring: str,
                        start: int, end: int) -> str:
        """Replaces a substring in a string."""
        return string[:start] + substring + string[end:]

    def _layers_to_printing_rows(self, width: int, positions: list[int],
                                 layers: list[_Layer]) -> list[str]:
        """Converts the layers into printing rows."""
        rows: list[str] = []
        for layer in layers:
            row = ' ' * width
            cur_col = 0

            for m_idx, m in enumerate(layer.morphisms):
                row = self._add_identities(rows, row, positions,
                                           cur_col, m.start)
                start_p, end_p = positions[m.start], positions[m.end]

                if m.morphism == _MorphismType.CUP:
                    m_str = self.chr_set["BOTTOM_L_CORNER"] + \
                            self.chr_set["LINE"] * (end_p - start_p - 1) + \
                            self.chr_set["BOTTOM_R_CORNER"]

                elif m.morphism == _MorphismType.SWAP:
                    m_str = self.chr_set["BOTTOM_L_CORNER"] + \
                            self.chr_set["TOP_R_CORNER"].\
                                center(end_p - start_p - 1,
                                       self.chr_set["LINE"]) + \
                            self.chr_set["BOTTOM_R_CORNER"]

                else:  # m.morphism == _MorphismType.SWAP_BOTTOM:
                    m_str = self.chr_set["TOP_L_CORNER"] + \
                            self.chr_set["BOTTOM_L_CORNER"].\
                                center(end_p - start_p - 1,
                                       self.chr_set["LINE"]) + \
                            self.chr_set["TOP_R_CORNER"]

                row = self._replace_substr(row, m_str, start_p, end_p + 1)

                stop_idx = len(positions) if m_idx == len(layer.morphisms) - 1 \
                    else layer.morphisms[m_idx + 1].start
                row = self._add_identities(rows, row, positions, m.end + 1,
                                           stop_idx)
                cur_col = m.end + 1

            rows.append(row)

        # If diagram contains no cups or swaps, just add identities
        if len(layers) == 0:
            row = ' ' * width
            for idx in range(len(positions)):
                row = self._replace_substr(row, self.chr_set["BAR"],
                                           positions[idx], positions[idx] + 1)
            rows = [row]

        return rows

    def diagram2str(self, diagram: Diagram) -> str:
        """Produces a string that contains a graphical representation of
        the input diagram using text characters. The diagram is expected
        to be in pregroup form, i.e. all words must precede the morphisms.

        Parameters
        ----------
        diagram: :py:class:`discopy.rigid.Diagram`
            The diagram to be printed.

        Returns
        -------
        str
            String that contains the graphical representation of the
            diagram.

        Raises
        ------
        ValueError
            If input is not a pregroup diagram.

        """

        if not is_pregroup_diagram(diagram):
            raise ValueError("The input is not a pregroup diagram.")

        word_space = ' ' * self.word_spacing

        # Collect words and their types
        header_data = [
            (box.name, str(box.cod))
            if self.discopy_types
            else (box.name, str(box.cod).replace(" @ ", self.chr_set["DOT"]))
            for box in diagram.boxes
            if isinstance(box, Word)
        ]

        # Create headers
        padded_words, underlines, padded_types = [], [], []
        for (wrd, typ) in header_data:
            pad_len = max(len(wrd), len(typ))
            padded_words.append(wrd.center(pad_len))
            underlines.append(self.chr_set["LINE"] * pad_len)
            padded_types.append(typ.center(pad_len))

        wrd_ln = word_space.join(padded_words)
        und_ln = word_space.join(underlines)
        typ_ln = word_space.join(padded_types)

        # Extract atomic types
        atomic_types = [str(t) for box in diagram.boxes for t in box.cod
                        if isinstance(box, Word)]

        # Find the position of each atomic type in the header
        pos = [typ_ln.find(atomic_types[0]) + len(atomic_types[0]) // 2]
        for t in atomic_types[1:]:
            idx = typ_ln.find(t, pos[-1] + 1)
            pos.append(idx + len(t) // 2)

        # Extract layers as given in the diagram (one cup/swap per layer)
        offsets = [(b, o) for (b, o)
                   in list(zip(diagram.boxes, diagram.offsets))
                   if isinstance(b, (Cup, Swap))]
        layers = self._induce_layers(atomic_types, offsets)

        # Compress if required
        if self.compress_layers:
            layers = self._compress_layers(layers)

        # Prepare printing rows
        print_rows = self._layers_to_printing_rows(len(und_ln), pos, layers)

        # Prepare the string
        diagram_str = wrd_ln + "\n" + und_ln + "\n" + typ_ln + "\n"
        for row in print_rows:
            diagram_str += row + "\n"

        return diagram_str
