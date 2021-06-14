import logging

from qiskit import IBMQ

import torch
from torch import nn
import pennylane as qml


#backend = 'ibmq_manila' # 5 qubits
#ibmqx_token = 'XXX'
#IBMQ.save_account(ibmqx_token, overwrite=True)
#IBMQ.load_account()


class QuonvLayer(nn.Module):
    def __init__(self, weights, stride=1, device="default.qubit", wires=4,
                 number_of_filters=1, circuit=None, filter_size=2, out_channels=4, seed=None, dtype=torch.float32):

        super(QuonvLayer, self).__init__()
        self.logger = logging.getLogger(__name__)
        if seed is not None:
            torch.manual_seed(seed)
        self.stride = stride
        self.wires = wires

        # setup device

        if device == "qulacs.simulator":
            self.device = qml.device(device, wires=self.wires, gpu=True)
        elif device == "qulacs.simulator-cpu":
            self.device = qml.device("qulacs.simulator", wires=self.wires, gpu=False)
        elif device == "qiskit.ibmq":
            # IBM quantum computer
            # define your credentials at top of this file
            # and uncomment the IBMQ account saving/loading
            self.device = qml.device('qiskit.ibmq', wires=self.wires, backend=backend)
        else:
            # default simulator
            self.device = qml.device(device, wires=self.wires)


        self.number_of_filters = number_of_filters
        self.filter_size = filter_size
        self.out_channels = out_channels
        self.dtype = dtype

        self.qlayer = qml.QNode(circuit, self.device, interface="torch", init_method=torch.nn.init.uniform_)
        if weights is not None:
            self.torch_qlayer = qml.qnn.TorchLayer(self.qlayer, weight_shapes={"weights": weights.shape},
                                              init_method=torch.nn.init.uniform_)
            self.torch_qlayer.weights.data = weights
        else:
            self.torch_qlayer = self.qlayer

    def convolve(self, img):
        bs, h, w, ch = img.size()
        #img.requires_grad = False
        """if ch > 1:
            img = img.mean(axis=-1).reshape(bs, h, w, 1)"""
        for b in range(bs):
            for j in range(0, h - self.filter_size + 1, self.stride):
                for k in range(0, w - self.filter_size + 1, self.stride):
                    # Process a squared nxn region of the image with a quantum circuit
                    yield img[b, j: j + self.filter_size, k: k + self.filter_size, :].flatten(), b, j, k

    def calc_out_dim(self, img):
        bs, h, w, ch = img.size()
        h_out = (int(h) - self.filter_size) // self.stride + 1
        w_out = (int(w) - self.filter_size) // self.stride + 1
        return bs, h_out, w_out, self.out_channels

    def forward(self, img):

        out = torch.empty(self.calc_out_dim(img), dtype=self.dtype)

        # print("debug", self.filter_size, self.stride, h,w,ch,h_out, out.shape)
        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for qnode_inputs, b, j, k in self.convolve(img):
            #print(qnode_inputs)
            q_results = self.torch_qlayer(
                qnode_inputs
            )

            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            out[b, j // self.stride, k // self.stride] = q_results

        return out

    def get_out_template(self, img):
        h, w = img.size()
        h_out = (h - self.filter_size) // self.stride + 1
        w_out = (w - self.filter_size) // self.stride + 1
        return torch.zeros(h_out, w_out)


class ExtractStatesQuonvLayer(QuonvLayer):

    def __init__(self, weights, stride=1, device="default.qubit", wires=4,
                 number_of_filters=1, circuit=None, filter_size=2, out_channels=4, seed=None, dtype=torch.complex64):
        super().__init__(weights, stride, device, wires, number_of_filters, circuit, filter_size, out_channels, seed, dtype)

    def calc_out_dim(self, img):
        bs, h, w, ch = img.size()
        h_out = (int(h) - self.filter_size) // self.stride + 1
        w_out = (int(w) - self.filter_size) // self.stride + 1
        return bs, h_out, w_out, 2**self.wires


"""class PreEncodedInputQuonvLayer(QuonvLayer):

    def calc_out_dim(self, img):
        return img.size

    def forward(self, img):

        out = torch.empty(self.calc_out_dim())

        for qnode_inputs, b, j, k in self.convolve(img):

            q_results = self.torch_qlayer(
                qnode_inputs
            )

            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            out[b, j // self.stride, k // self.stride] = q_results"""

