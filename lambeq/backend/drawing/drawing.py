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
Lambeq drawing
==============
Functionality for drawing lambeq diagrams. This work is based on DisCoPy
(https://discopy.org/) which is released under the BSD 3-Clause "New"
or "Revised" License.

"""

from __future__ import annotations

from math import sqrt
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING

from PIL import Image

from lambeq.backend import grammar
from lambeq.backend.drawing.drawable import (BoxNode, DrawableDiagram,
                                             DrawablePregroup,
                                             WireEndpointType)
from lambeq.backend.drawing.drawing_backend import (DEFAULT_ASPECT,
                                                    DEFAULT_MARGINS,
                                                    DrawingBackend)
from lambeq.backend.drawing.helpers import drawn_as_spider, needs_asymmetry
from lambeq.backend.drawing.mat_backend import MatBackend
from lambeq.backend.drawing.tikz_backend import TikzBackend
from lambeq.backend.grammar import Diagram


if TYPE_CHECKING:
    from IPython.core.display import HTML as HTML_ty


def draw(diagram: Diagram, **params) -> None:
    """
    Draws a grammar diagram.

    Parameters
    ----------
    diagram: Diagram
        Diagram to draw.
    draw_as_nodes : bool, optional
        Whether to draw boxes as nodes, default is `False`.
    color : string, optional
        Color of the box or node, default is white (`'#ffffff'`) for
        boxes and red (`'#ff0000'`) for nodes.
    textpad : pair of floats, optional
        Padding between text and wires, default is `(0.1, 0.1)`.
    draw_type_labels : bool, optional
        Whether to draw type labels, default is `False`.
    draw_box_labels : bool, optional
        Whether to draw box labels, default is `True`.
    aspect : string, optional
        Aspect ratio, one of `['auto', 'equal']`.
    margins : tuple, optional
        Margins, default is `(0.05, 0.05)`.
    nodesize : float, optional
        BoxNode size for spiders and controlled gates.
    fontsize : int, optional
        Font size for the boxes, default is `12`.
    fontsize_types : int, optional
        Font size for the types, default is `12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call `plt.show()`.
    to_tikz : bool, optional
        Whether to output tikz code instead of matplotlib.
    asymmetry : float, optional
        Make a box and its dagger mirror images, default is
        `.25 * any(box.is_dagger for box in diagram.boxes)`.

    """

    params['asymmetry'] = params.get(
        'asymmetry', .25 * needs_asymmetry(diagram))

    drawable = DrawableDiagram.from_diagram(diagram)
    drawable.scale_and_pad(params.get('scale', (1, 1)),
                           params.get('pad', (0, 0)))

    if 'backend' in params:
        backend: DrawingBackend = params.pop('backend')
    elif params.get('to_tikz', False):
        backend = TikzBackend(
            use_tikzstyles=params.get('use_tikzstyles', None))
    else:
        backend = MatBackend(figsize=params.get('figsize', None))

    min_size = 0.01
    max_v = max([v for point in ([point.coordinates for point in
                 drawable.wire_endpoints + drawable.boxes]) for v in point]
                + [min_size])

    params['nodesize'] = round(params.get('nodesize', 1.) / sqrt(max_v), 3)

    backend = _draw_wires(backend, drawable, **params)
    backend.draw_spiders(drawable, **params)

    for node in drawable.boxes:
        if not drawn_as_spider(node.obj):
            backend = _draw_box(backend, drawable, node, **params)

    backend.output(
        path=params.get('path', None),
        baseline=len(drawable.boxes) / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT_MARGINS),
        aspect=params.get('aspect', DEFAULT_ASPECT))


def draw_pregroup(diagram: Diagram, **params) -> None:
    """
    Draws a pregroup grammar diagram.
    A pregroup diagram is structured as:
        (State @ State ... State) >> (Cups and Swaps)

    Parameters
    ----------
    diagram: Diagram
        Diagram to draw.
    draw_as_nodes : bool, optional
        Whether to draw boxes as nodes, default is `False`.
    color : string, optional
        Color of the box or node, default is white (`'#ffffff'`) for
        boxes and red (`'#ff0000'`) for nodes.
    textpad : pair of floats, optional
        Padding between text and wires, default is `(0.1, 0.1)`.
    aspect : string, optional
        Aspect ratio, one of `['auto', 'equal']`.
    margins : tuple, optional
        Margins, default is `(0.05, 0.05)`.
    fontsize : int, optional
        Font size for the boxes, default is `12`.
    fontsize_types : int, optional
        Font size for the types, default is `12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call `plt.show()`.
    to_tikz : bool, optional
        Whether to output tikz code instead of matplotlib.

    """

    drawable = DrawablePregroup.from_diagram(diagram)
    drawable.scale_and_pad(params.get('scale', (1, 1)),
                           params.get('pad', (0, 0)))

    if 'backend' in params:
        backend: DrawingBackend = params.pop('backend')
    elif params.get('to_tikz', False):
        backend = TikzBackend(
            use_tikzstyles=params.get('use_tikzstyles', None))
    else:
        backend = MatBackend(figsize=params.get('figsize', None))

    backend = _draw_wires(backend, drawable, **params)
    backend.draw_spiders(drawable, **params)

    for node in drawable.boxes:
        if not drawn_as_spider(node.obj):
            backend = _draw_pregroup_state(backend, node, **params)

    backend.output(
        path=params.get('path', None),
        baseline=len(drawable.boxes) / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT_MARGINS),
        aspect=params.get('aspect', DEFAULT_ASPECT))


def to_gif(diagrams: list[Diagram],
           path: str | None = None,
           timestep: int = 500,
           loop: bool = False,
           **params) -> str | HTML_ty:
    """
    Builds a GIF stepping through the given diagrams.

    Parameters
    ----------
    diagrams: list of Diagrams
        Sequence of diagrams to draw.
    path : str
        Where to save the image, if `None` a gif gets created.
    timestep : int, optional
        Time step in milliseconds, default is `500`.
    loop : bool, optional
        Whether to loop, default is `False`
    params : any, optional
        Passed to `Diagram.draw`.

    Returns
    -------
    IPython.display.HTML or str
        HTML to display the generated GIF

    """

    steps, frames = diagrams, []
    path = path or os.path.basename(NamedTemporaryFile(
        suffix='.gif', prefix='tmp_', dir='.').name)

    with TemporaryDirectory() as directory:
        for i, _diagram in enumerate(steps):
            tmp_path = os.path.join(directory, f'{i}.png')
            _diagram.draw(path=tmp_path, **params)
            frames.append(Image.open(tmp_path))

        if loop:
            frames = frames + frames[::-1]

        frames[0].save(path, format='GIF', append_images=frames[1:],
                       save_all=True, duration=timestep,
                       **{'loop': 0} if loop else {})  # type: ignore[arg-type]

        try:
            from IPython.display import HTML
            return HTML(f'<img src="{path}">')
        except ImportError:
            return f'<img src="{path}">'


def draw_equation(*terms: grammar.Diagram,
                  symbol: str = '=',
                  space: float = 1,
                  path: str | None = None,
                  **params) -> None:
    """
    Draws an equation with multiple diagrams.

    Parameters
    ----------
    terms: list of Diagrams
        Diagrams in equation.
    symbol: str
        Symbol separating equations. '=' by default.
    space: float
        Amount of space between adjacent diagrams.
    path : str, optional
        Where to save the image, if `None` we call `plt.show()`.
    **params:
        Additional drawing parameters, passed to :meth:`draw`.

    """

    def height(term):
        if hasattr(term, 'terms'):
            return max(height(d) for d in term.terms)
        return len(term) or 1

    params['asymmetry'] = params.get(
        'asymmetry', .25 * any(needs_asymmetry(d) for d in terms))

    max_height = max(map(height, terms))
    pad = params.get('pad', (0, 0))
    scale_x, scale_y = params.get('scale', (1, 1))

    if 'backend' in params:
        backend: DrawingBackend = params.pop('backend')
    elif params.get('to_tikz', False):
        backend = TikzBackend(
            use_tikzstyles=params.get('use_tikzstyles', None))
    else:
        backend = MatBackend(figsize=params.get('figsize', None))

    for i, term in enumerate(terms):
        scale = (scale_x, scale_y * max_height / height(term))
        term.draw(**dict(
            params, show=False, path=None,
            backend=backend, scale=scale, pad=pad))
        pad = (backend.max_width + space, 0)
        if i < len(terms) - 1:
            backend.draw_text(symbol, pad[0], scale_y * max_height / 2)
            pad = (pad[0] + space, pad[1])

    return backend.output(
        path=path,
        baseline=max_height / 2,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT_MARGINS),
        aspect=params.get('aspect', DEFAULT_ASPECT))


def _draw_box(backend: DrawingBackend,
              drawable_diagram: DrawableDiagram,
              drawable_box: BoxNode,
              asymmetry: float,
              **params) -> DrawingBackend:
    """
    Draws a box on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    drawable_box: BoxNode
        A BoxNode to be drawn. Must be in `drawable_diagram`.
    asymmetry: float
        Amount of asymmetry, used to represent transposes,
        conjugates and daggers,
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    box = drawable_box.obj

    if not box.dom and not box.cod:
        left, right = drawable_box.x, drawable_box.x

    all_wires_pos = [drawable_diagram.wire_endpoints[wire].x
                     for wire in
                     drawable_box.cod_wires + drawable_box.dom_wires]

    if not all_wires_pos:
        all_wires_pos = [drawable_box.x]

    left = min(all_wires_pos) - 0.25
    right = max(all_wires_pos) + 0.25
    height = drawable_box.y - .25

    points = [[left, height], [right, height],
              [right, height + .5], [left, height + .5]]

    # TODO: Update once this functionality is added to grammar
    is_conjugate = getattr(box, 'is_conjugate', False)
    is_transpose = getattr(box, 'is_transpose', False)

    if is_transpose:
        points[0][0] -= asymmetry
    elif is_conjugate:
        points[3][0] -= asymmetry
    elif isinstance(box, grammar.Daggered):
        points[1][0] += asymmetry
    else:
        points[2][0] += asymmetry

    backend.draw_polygon(*points)

    if params.get('draw_box_labels', True):
        backend.draw_text(box.name, drawable_box.x, drawable_box.y,
                          ha='center', va='center',
                          fontsize=params.get('fontsize', None))

    return backend


def _draw_pregroup_state(backend: DrawingBackend,
                         drawable_box: BoxNode,
                         **params) -> DrawingBackend:
    """
    Draws a pregroup word state on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_box: BoxNode
        A BoxNode to be drawn.
    **params:
        Additional drawing parameters. See `drawing.draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the box's graphic.

    """

    box = drawable_box.obj

    left = drawable_box.x
    right = left + 2
    height = drawable_box.y - .25

    points = [[left, height], [right, height],
              [right, height + .5], [(left + right) / 2, height + 0.6],
              [left, height + .5]]

    backend.draw_polygon(*points)
    backend.draw_text(box.name, drawable_box.x + 1, drawable_box.y,
                      ha='center', va='center',
                      fontsize=params.get('fontsize', None))

    return backend


def _draw_wires(backend: DrawingBackend,
                drawable_diagram: DrawableDiagram,
                **params) -> DrawingBackend:
    """
    Draws all wires of a diagram on a given backend.

    Parameters
    ----------
    backend: DrawingBackend
        A lambeq drawing backend.
    drawable_diagram: DrawableDiagram
        A drawable diagram.
    **params:
        Additional drawing parameters. See :meth:`draw`.

    Returns
    -------
    backend: DrawingBackend
        Drawing backend updated with the wires' graphic.

    """

    for src_idx, tgt_idx in drawable_diagram.wires:
        source = drawable_diagram.wire_endpoints[src_idx]
        target = drawable_diagram.wire_endpoints[tgt_idx]

        backend.draw_wire(
            source.coordinates, target.coordinates)

        if (params.get('draw_type_labels', True) and source.kind in
                {WireEndpointType.INPUT, WireEndpointType.COD}):

            i, j = source.coordinates
            pad_i, pad_j = params.get('textpad', (.1, .1))
            pad_j = 0 if source.kind == WireEndpointType.INPUT else pad_j
            backend.draw_text(
                str(source.obj), i + pad_i, j - pad_j,
                fontsize=params.get('fontsize_types',
                                    params.get('fontsize', None)),
                verticalalignment='top')
    return backend
