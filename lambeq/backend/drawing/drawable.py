# Copyright 2021-2023 Cambridge Quantum Computing Ltd.
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
Drawable Components
===================
Utilities to convert a grammar diagram into a drawable form.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Self

from lambeq.backend import grammar


class WireEndpointType(Enum):
    """An enumeration for :py:class:`WireEndpoint`.

    WireEndpoints in diagrams can be of 4 types:

    .. glossary::

        DOM
            Domain of a box.

        COD
            Codomain of a box.

        INPUT
            Input wire to the diagram.

        OUTPUT
            Output wire from the diagram.

    """

    DOM = 0
    COD = 1
    INPUT = 2
    OUTPUT = 3

    def __repr__(self) -> str:
        return self.name


@dataclass
class WireEndpoint:
    """
    One end of a wire in a DrawableDiagram.

    Attributes
    ----------
    kind: WireEndpointType
        Type of wire endpoint.
    obj: grammar.Ty
        Categorial type carried by the wire.
    x: float
        X coordinate of the wire end.
    y: float
        Y coordinate of the wire end.
    coordinates: (float, float)
        (x, y) coordinates.

    """

    kind: WireEndpointType
    obj: grammar.Ty

    x: float
    y: float

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class BoxNode:
    """
    Box in a DrawableDiagram.

    Attributes
    ----------
    obj: grammar.Box
        Grammar box represented by the node.
    x: float
        X coordinate of the box.
    y: float
        Y coordinate of the box.
    coordinates: (float, float)
        (x, y) coordinates.
    dom_wires: list of int
        Wire endpoints in the domain of the box, represented by
        indices into an array maintained by `DrawableDiagram`.
    com_wires: list of int
        Wire endpoints in the codomain of the box, represented by
        indices into an array maintained by `DrawableDiagram`.

    """

    obj: grammar.Box

    x: float
    y: float

    dom_wires: list[int] = field(default_factory=list)
    cod_wires: list[int] = field(default_factory=list)

    @property
    def coordinates(self):
        return (self.x, self.y)

    def add_dom_wire(self, idx: int) -> None:
        """
        Add a wire to to box's domain.

        Parameters
        ----------
        idx : int
            Index of wire in associated `DrawableDiagram`'s
            `wire_endpoints` attribute.

        """
        self.dom_wires.append(idx)

    def add_cod_wire(self, idx: int) -> None:
        """
        Add a wire to to box's codomain.

        Parameters
        ----------
        idx : int
            Index of wire in associated `DrawableDiagram`'s
            `wire_endpoints` attribute.

        """
        self.cod_wires.append(idx)


@dataclass
class DrawableDiagram:
    """
    Representation of a lambeq diagram carrying all
    information necessary to render it.

    Attributes
    ----------
    boxes: list of BoxNode
        Boxes in the diagram.
    wire_endpoints: list of WireEndpoint
        Endpoints for all wires in the diagram.
    wires: list of tuple of the form (int, int)
        The wires in a diagram, each represented by the indices of
        its 2 endpoints in `wire_endpoints`.

    """

    boxes: list[BoxNode] = field(default_factory=list)
    wire_endpoints: list[WireEndpoint] = field(default_factory=list)
    wires: list[tuple[int, int]] = field(default_factory=list)

    def _add_wire(self,
                  source: int,
                  target: int) -> None:
        """Add an edge between 2 connected wire endpoints."""

        self.wires.append((source, target))

    def _add_wire_end(self, wire_end: WireEndpoint) -> int:
        """Add a `WireEndpoint` to the diagram."""

        self.wire_endpoints.append(wire_end)
        return len(self.wire_endpoints) - 1

    def _add_boxnode(self, box: BoxNode) -> int:
        """Add a `BoxNode` to the diagram."""

        self.boxes.append(box)
        return len(self.boxes) - 1

    def _add_box(self,
                 scan: list[int],
                 box: grammar.Box,
                 off: int,
                 depth: int,
                 x_pos: float,
                 max_depth: float) -> list[int]:
        """Add a box to the graph, creating necessary wire endpoints."""

        node = BoxNode(box, x_pos, max_depth - depth - .5)

        self._add_boxnode(node)

        # Create a node representing each element in the box's domain
        for i, obj in enumerate(box.dom):
            nbr_idx = scan[off + i]
            wire_end = WireEndpoint(WireEndpointType.DOM,
                                    obj=obj,
                                    x=self.wire_endpoints[nbr_idx].x,
                                    y=max_depth - depth - .25)

            wire_idx = self._add_wire_end(wire_end)
            node.add_dom_wire(wire_idx)
            self._add_wire(nbr_idx, wire_idx)

        scan_insert = []

        # Create a node representing each element in the box's codomain
        for i, obj in enumerate(box.cod):
            x = x_pos - len(box.cod[1:]) / 2 + i
            y = max_depth - depth - .75

            wire_end = WireEndpoint(WireEndpointType.COD,
                                    obj=obj,
                                    x=x,
                                    y=y)

            wire_idx = self._add_wire_end(wire_end)
            scan_insert.append(wire_idx)
            node.add_cod_wire(wire_idx)

        # Replace node's dom with its cod in scan
        return scan[:off] + scan_insert + scan[off + len(box.dom):]

    def _make_space(self,
                    scan: list[int],
                    box: grammar.Box,
                    off: int) -> float:
        """Determines x coord for a new box.
        Modifies x coordinates of existing nodes to make space."""

        if not scan:
            return 0

        half_width = len(box.cod[:-1]) / 2 + 1

        if not box.dom:
            if not off:
                x = self.wire_endpoints[scan[0]].x - half_width
            elif off == len(scan):
                x = self.wire_endpoints[scan[-1]].x + half_width
            else:
                right = self.wire_endpoints[scan[off + len(box.dom)]].x
                x = (self.wire_endpoints[scan[off - 1]].x + right) / 2
        else:
            right = self.wire_endpoints[scan[off + len(box.dom) - 1]].x
            x = (self.wire_endpoints[scan[off]].x + right) / 2

        if off and self.wire_endpoints[scan[off - 1]].x > x - half_width:
            limit = self.wire_endpoints[scan[off - 1]].x
            pad = limit - x + half_width

            for node in self.boxes + self.wire_endpoints:
                if node.x <= limit:
                    node.x -= pad

        if (off + len(box.dom) < len(scan)
                and (self.wire_endpoints[scan[off + len(box.dom)]].x
                     < x + half_width)):
            limit = self.wire_endpoints[scan[off + len(box.dom)]].x
            pad = x + half_width - limit

            for node in self.boxes + self.wire_endpoints:
                if node.x >= limit:
                    node.x += pad

        return x

    @classmethod
    def from_diagram(cls, diagram: grammar.Diagram) -> Self:
        """
        Builds a graph representation of the diagram, calculating
        coordinates for each box and wire.

        Parameters
        ----------
        diagram : grammar Diagram
            A lambeq diagram.

        Returns
        -------
        drawable : DrawableDiagram
            Representation of diagram including all coordinates
            necessary to draw it.

        """

        drawable = cls()

        scan = []

        for i, obj in enumerate(diagram.dom):
            wire_end = WireEndpoint(WireEndpointType.INPUT,
                                    obj=obj,
                                    x=i,
                                    y=len(diagram) or 1)
            wire_end_idx = drawable._add_wire_end(wire_end)
            scan.append(wire_end_idx)

        for depth, (box, off) in enumerate(zip(diagram.boxes,
                                               diagram.offsets)):
            x = drawable._make_space(scan, box, off)
            scan = drawable._add_box(scan, box, off, depth, x, len(diagram))

        for i, obj in enumerate(diagram.cod):
            wire_end = WireEndpoint(WireEndpointType.OUTPUT,
                                    obj=obj,
                                    x=drawable.wire_endpoints[scan[i]].x,
                                    y=0)
            wire_end_idx = drawable._add_wire_end(wire_end)
            drawable._add_wire(scan[i], wire_end_idx)

        # Set the min x and y coordinates to 0, to make scaling easier
        # to reason about.
        min_x = min(
            [node.x for node in drawable.boxes + drawable.wire_endpoints])
        min_y = min(
            [node.y for node in drawable.boxes + drawable.wire_endpoints])

        for node in drawable.boxes + drawable.wire_endpoints:
            node.x -= min_x
            node.y -= min_y

        return drawable

    def scale_and_pad(self,
                      scale: tuple[float, float],
                      pad: tuple[float, float]):
        """Scales and pads the diagram as specified.

        Parameters
        ----------
        scale : tuple of 2 floats
            Scaling factors for x and y axes respectively.
        pad : tuple of 2 floats
            Padding values for x and y axes respectively.

        """

        min_x = min([node.x for node in self.boxes + self.wire_endpoints])
        min_y = min([node.y for node in self.boxes + self.wire_endpoints])

        for wire_end in self.wire_endpoints:
            wire_end.x = min_x + (wire_end.x - min_x) * scale[0] + pad[0]
            wire_end.y = min_y + (wire_end.y - min_y) * scale[1] + pad[1]

        for box in self.boxes:
            box.x = min_x + (box.x - min_x) * scale[0] + pad[0]
            box.y = min_y + (box.y - min_y) * scale[1] + pad[1]

            for wire_end_idx in box.dom_wires:
                self.wire_endpoints[wire_end_idx].y = box.y + 0.25

            for wire_end_idx in box.cod_wires:
                self.wire_endpoints[wire_end_idx].y = box.y - 0.25
