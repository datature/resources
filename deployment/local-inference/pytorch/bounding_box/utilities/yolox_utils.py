# pylint: disable=E0401, R0201
"""Converter tools needed for PyTorch model conversion and prediction"""
import torch
from torch import nn
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxArgMax(nn.Module, OnnxToTorchModule):

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        return torch.argmax(input_data, dim=-1, keepdim=True)  # pylint: disable=E1101

@add_converter(operation_type='ArgMax', version=11)
@add_converter(operation_type='ArgMax', version=12)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxArgMax(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
