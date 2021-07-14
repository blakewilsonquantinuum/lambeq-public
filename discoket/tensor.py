__all__ = ['TensorDiagram']

import dataclasses
from typing import Any, Callable, List, Optional, Type

from discopy.rigid import Box, Cup, Diagram, Spider
import numpy as np
import tensornetwork as tn

# Types
Tensor = Any


@dataclasses.dataclass
class TensorNetwork:
    nodes: List[tn.AbstractNode]
    dangling_edges: List[tn.Edge]


class TensorDiagram:
    def __init__(self, diagram: Diagram) -> None:
        self.diagram = diagram

    def get_network(self,
                    closure: Callable[[int, Box, Diagram], Tensor],
                    dtype: Optional[Type[np.number]] = None) -> TensorNetwork:
        nodes = []
        scan: List[tn.Edge] = []
        for i, (box, offset) in enumerate(zip(self.diagram.boxes,
                                              self.diagram.offsets)):
            if isinstance(box, Cup):
                tn.connect(scan[offset], scan[offset+1])
                del scan[offset:offset+2]
                continue

            if isinstance(box, Spider):
                assert box.dom
                if dtype is None:
                    raise ValueError('`dtype` could not be inferred, please '
                                     'pass it explicitly')
                node: tn.AbstractNode = tn.CopyNode(len(box.dom)+len(box.cod),
                                                    scan[offset].dimension,
                                                    dtype=dtype)
                for i in range(len(box.dom)):
                    tn.connect(scan[offset+i], node[i])
            else:
                tensor = closure(i, box, self.diagram)
                if dtype is None:
                    dtype = tensor.detach().cpu().numpy().dtype
                node = tn.Node(tensor)
            scan[offset:offset+len(box.dom)] = node[len(box.dom):]
            nodes.append(node)

        return TensorNetwork(nodes=nodes, dangling_edges=scan)

    def __call__(self,
                 closure: Callable[[int, Box, Diagram], Tensor],
                 dtype: Optional[Type[np.number]] = None) -> Tensor:
        nwk = self.get_network(closure)
        return tn.contractors.auto(nwk.nodes, nwk.dangling_edges).tensor
