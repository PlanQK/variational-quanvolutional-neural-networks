from circuitcomponents import CircuitComponents
import pennylane as qml
from pennylane import numpy as np
import torch


class FRQI_for_2x2(CircuitComponents):
    """ The following code is heavily inspired by https://www.cs.umd.edu/class/fall2018/cmsc657/projects/group_6.pdf
        and P. Q. Le, F. Dong, und K. Hirota, „A flexible representation of quantum images for polynomial preparation,
        image compression, and processing operations“, Quantum Inf Process, Bd. 10, Nr. 1, S. 63–84, Feb. 2011,
        doi: 10.1007/s11128-010-0177-y. page 69

         Input is expected to be a flat tensor with 4 entries with values in [0,1].
    """

    def __init__(self, filter_length=2):
        if not filter_length == 2:
            raise Exception(f"This Encoding needs filter_length to be 2 not {filter_length}.")
        self.name = "FRQI for 2x2 images"
        self.n_qubits = 3
        self.required_qubits = self.n_qubits

    def img_to_theta(self, img):
        """ normalized rgb image into [0, pi/2] """

        # normalize image
        # img = img / 255.0

        # [0,1] -> [0, pi/2]
        img = torch.asin(img)

        # shape(n,n) -> shape(n*n)
        # img = img.flatten()

        return img

    def circuit(self, inputs):

        angles = self.img_to_theta(inputs)
        qubits = list(range(self.n_qubits))

        ### ENCODING ###

        # apply hadamard gates to each qubit
        for qubit in qubits[:-1]:
            qml.Hadamard(wires=qubit)

        for i, theta in enumerate(angles):

            # flip bits to encode pixel position
            qml.PauliX(qubits[0])

            if i % 2 == 0:  # = first in a row
                qml.PauliX(qubits[1])

            qml.CRY(theta, wires=[qubits[0], qubits[2]])
            qml.CNOT(wires=[qubits[0], qubits[1]])
            qml.CRY(-theta, wires=[qubits[1], qubits[2]])
            qml.CNOT(wires=[qubits[0], qubits[1]])
            qml.CRY(theta, wires=[qubits[1], qubits[2]])


class FRQI_for_4x4(CircuitComponents):
    """ The following code is heavily inspired by https://www.cs.umd.edu/class/fall2018/cmsc657/projects/group_6.pdf
        and P. Q. Le, F. Dong, und K. Hirota, „A flexible representation of quantum images for polynomial preparation,
        image compression, and processing operations“, Quantum Inf Process, Bd. 10, Nr. 1, S. 63–84, Feb. 2011,
        doi: 10.1007/s11128-010-0177-y. page 69

        Input is expected to be a flat tensor with 16 entries with values in [0,1].
    """

    def __init__(self, filter_length=4):
        if not filter_length == 4:
            raise Exception(f"This Encoding needs filter_length to be 4 not {filter_length}.")
        self.name = "FRQI for 4x4 images"
        self.n_qubits = 8
        self.qubits = list(range(self.n_qubits))
        self.control_qubits = self.qubits[:4]
        self.work_qubits = self.qubits[4:7]
        self.color_qubit = self.qubits[7]
        self.required_qubits = self.n_qubits

    def img_to_theta(self, img):
        """ normalized rgb image into [0, pi/2] """

        # normalize image
        # img = img / 255.0

        # [0,1] -> [0, pi/2]
        img = torch.asin(img)

        # shape(n,n) -> shape(n*n)
        # img = img.flatten()

        return img

    def bitstring_to_numpy(self, bitstring, reverse=False):
        """
        Converts a bitstring to boolean numpy array
        """

        # result = np.zeros(len(bitstring))

        if reverse:
            bitstring = reversed(bitstring)

        result = np.array(list(bitstring)).astype(bool)

        return result

    def circuit(self, inputs):

        angles = self.img_to_theta(inputs)

        ### ENCODING ###

        # apply hadamard gates to each qubit
        for qubit in self.control_qubits:
            qml.Hadamard(wires=qubit)

        last_bitstring = self.bitstring_to_numpy('0000')

        for i, theta in enumerate(angles.flatten()):

            # encode pixel position
            binary_i = self.bitstring_to_numpy(format(15 - i, "b").zfill(4), reverse=True)
            changed_bits = np.logical_xor(binary_i, last_bitstring)
            last_bitstring = binary_i

            for p, flipped in enumerate(changed_bits):
                if flipped:
                    qml.PauliX(wires=self.control_qubits[p])

            # controled rotations
            qml.Toffoli(wires=[self.control_qubits[0], self.control_qubits[1], self.work_qubits[0]])
            qml.Toffoli(wires=[self.control_qubits[2], self.work_qubits[0], self.work_qubits[1]])
            qml.Toffoli(wires=[self.control_qubits[3], self.work_qubits[1], self.work_qubits[2]])

            qml.CRY(2 * theta, wires=[self.work_qubits[2], self.color_qubit])

            qml.Toffoli(wires=[self.control_qubits[3], self.work_qubits[1], self.work_qubits[2]])
            qml.Toffoli(wires=[self.control_qubits[2], self.work_qubits[0], self.work_qubits[1]])
            qml.Toffoli(wires=[self.control_qubits[0], self.control_qubits[1], self.work_qubits[0]])
