"""
Generates a tiny, frozen linear+ReLU ONNX model with fixed (seeded) random
weights, standing in for a real feature-extractor backbone. Built directly
via onnx.helper (no PyTorch dependency) so it can be regenerated without a
training framework - only `pip install onnx numpy` is needed.

Input "input": [batch, INPUT_DIM] float32
Output "embedding": [batch, OUTPUT_DIM] float32, computed as relu(input @ W + b)
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

INPUT_DIM = 8
OUTPUT_DIM = 8
SEED = 42

rng = np.random.default_rng(SEED)
W = rng.normal(scale=0.3, size=(INPUT_DIM, OUTPUT_DIM)).astype(np.float32)
b = rng.normal(scale=0.1, size=(OUTPUT_DIM,)).astype(np.float32)

W_init = helper.make_tensor('W', TensorProto.FLOAT, W.shape, W.flatten().tolist())
b_init = helper.make_tensor('b', TensorProto.FLOAT, b.shape, b.flatten().tolist())

input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', INPUT_DIM])
output_tensor = helper.make_tensor_value_info('embedding', TensorProto.FLOAT, ['batch', OUTPUT_DIM])

matmul_node = helper.make_node('MatMul', ['input', 'W'], ['matmul_out'])
add_node = helper.make_node('Add', ['matmul_out', 'b'], ['add_out'])
relu_node = helper.make_node('Relu', ['add_out'], ['embedding'])

graph = helper.make_graph(
    [matmul_node, add_node, relu_node],
    'nexus_ppfl_linear_relu_backbone',
    [input_tensor],
    [output_tensor],
    initializer=[W_init, b_init],
)

model = helper.make_model(
    graph,
    producer_name='nexus-ppfl-fixture-generator',
    opset_imports=[helper.make_opsetid('', 13)],
)
onnx.checker.check_model(model)
onnx.save(model, 'model.onnx')
print('saved model.onnx')
