import numpy as np
import torch
from torch import nn

from models.quonv_layer import QuonvLayer
from utils.circuitcomponents_utils import generate_corresponding_circuit, get_wires_number


# from utils.circuitcomponents_utils import


class QNNModel(nn.Module):
    """Quantum Trainable Model"""

    def __init__(self, weights, out_channels, circuit, out_features, n_rotations=4, wires=4,
                 seed=None, stride=2, filter_size=2, img_size=28, device="default.qubit"):
        super().__init__()

        self.qlayer_1 = QuonvLayer(stride=stride,
                                   circuit=circuit,
                                   weights=weights,
                                   wires=wires,
                                   out_channels=out_channels,
                                   seed=seed,
                                   filter_size=filter_size,
                                   device=device)

        self.flatten = nn.Flatten()

        n_in_parameter = (img_size - filter_size) // stride + 1
        self.linear = nn.Linear(in_features=n_in_parameter * n_in_parameter * out_channels, out_features=out_features)

    def forward(self, input):
        x = self.qlayer_1.forward(input)
        x = self.flatten(x)
        return self.linear(x)

    def draw_qlayer_circuit(self):
        print(self.qlayer_1.qlayer.draw())
        print("---------------------------------------")


def initialize_QNN_model(params, device, encoding=True):
    weights_shape = (params['circuit_layers'], params['n_rotations'])
    weights = np.random.default_rng(params['weights_seed']).uniform(-1, 1, weights_shape)
    weights = torch.tensor(weights)
    circuit = generate_corresponding_circuit(params, weights_initialized=weights, encoding=encoding)
    if encoding:
        stride = params['stride']
        filter_size = params['filter_length']
        img_size = params['img_size']
    else:
        stride = 1
        filter_size = 1
        img_size = (params['img_size'] - params['filter_length']) // params['stride'] + 1

    model = QNNModel(
        weights=weights,
        circuit=circuit,
        out_channels=params['out_channels'],
        out_features=params['out_features'],
        wires=get_wires_number(params),
        seed=params["weights_seed"],
        filter_size=filter_size,
        stride=stride,
        img_size=img_size,
        device=device
    )

    return model, circuit
