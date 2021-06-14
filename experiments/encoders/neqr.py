import pennylane as qml
from pennylane import numpy as np
from pyeda.inter import *
#import qiskit as qis
from circuitcomponents import CircuitComponents


class NEQR(CircuitComponents):
    """ Encode a 2^n x 2^n picture with 2^q values in q + 2n qubits.
        Use the espresso algorithm to optimize the logic expressions that are derived from
        the combinations of Toffoli-Gates (basically apply PauliX to the target, if XOr(inputs) == True).
        If this encoding is used the input needs to be an image with pixel values in 0-255 but the dtype should
        be float because Pytorch returns the same dtype as the input.
        This method is described in the Zhang et al paper:
        NEQR: a novel enhanced quantum representation of digital images

        Input is expected to be a (filter_length x filter_length) tensor with values in [0,1],
        where filter_length == 2**n for some n.
    """

    def __init__(self, q=8, filter_length=2):
        if not 2 ** int(np.log2(filter_length)) == filter_length:
            raise Exception ("filter_length needs to be equal to 2**n for some integer n.")
        self.q = q
        self.n = int(np.log2(filter_length))

        # Qiskit transpile not compatible with newer PennyLane version so we've done this step manually below.
        '''self.mcx_circuits = dict()
        for n in range(2, 2 * self.n + 1):
            self.mcx_circuits[n] = self.mcx(n)'''
        self.mcx_circuits = {2: qml.Toffoli,
                             3: self.toff3,
                             4: self.toff4}

        self.qubits_ev = list(map(exprvar, [f"b{i}" for i in range(self.q + 2 * self.n)]))
        self.not_qubits_ev = [f"~{i}" for i in self.qubits_ev]

        self.name = "NEQR"

        self.required_qubits = self.q + 2 * self.n

    # Functions needed to construct the circuit.
    def wire(self, log_var):
        """ return the wire of a logical variable. """
        return int(str(log_var.top)[1:])

    def toff4(self, wires):
        """ Generated roughly as follows:
            circ.mcx(..., ..., ..., mode="noancilla")
            circ = qis.transpile(circ, basis_gates=["h", "u1", "id, " "crx", "crz", "rx", "rz", "cx", "x", "z", "ccx"],
                          optimization_level=3)
            def func():
                qml.from_qiskit(circ)(wires=...)
                return(qml.expval(qml.PauliZ(0)))
            node = qml.QNode(func, dev)
            node.circuit.operations_in_order"""
        control0, control1, control2, control3 = wires[:-1]
        target = wires[-1]
        qml.PhaseShift(0.19634954084936207, wires=[control3])

        qml.Hadamard(wires=[target])
        qml.CNOT(wires=[control3, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control3, target])

        qml.CNOT(wires=[control3, control2])
        qml.PhaseShift(-0.19634954084936207, wires=[control2])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control2, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control2, target])
        qml.CNOT(wires=[control3, control2])
        qml.PhaseShift(0.19634954084936207, wires=[control2])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control2, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control2, target])

        qml.CNOT(wires=[control2, control1])
        qml.PhaseShift(-0.19634954084936207, wires=[control1])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.CNOT(wires=[control3, control1])
        qml.PhaseShift(0.19634954084936207, wires=[control1])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.CNOT(wires=[control2, control1])
        qml.PhaseShift(-0.19634954084936207, wires=[control1])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.CNOT(wires=[control3, control1])
        qml.PhaseShift(0.19634954084936207, wires=[control1])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control1, target])

        qml.CNOT(wires=[control1, control0])
        qml.PhaseShift(-0.19634954084936207, wires=[control0])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control3, control0])
        qml.PhaseShift(0.19634954084936207, wires=[control0])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control2, control0])
        qml.PhaseShift(-0.19634954084936207, wires=[control0])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control3, control0])
        qml.PhaseShift(0.19634954084936207, wires=[control0])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control1, control0])
        qml.PhaseShift(-0.19634954084936207, wires=[control0])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control3, control0])
        qml.PhaseShift(0.19634954084936207, wires=[control0])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control2, control0])
        qml.PhaseShift(-0.19634954084936207, wires=[control0])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control3, control0])
        qml.PhaseShift(0.19634954084936207, wires=[control0])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(-0.19634954084936207, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.19634954084936207, wires=[target])
        qml.Hadamard(wires=[target])

    def toff3(self, wires):
        """ Generated roughly as follows:
            circ.mcx(..., ..., ..., mode="noancilla")
            circ = qis.transpile(circ, basis_gates=["h", "u1", "id, " "crx", "crz", "rx", "rz", "cx", "x", "z", "ccx"],
                          optimization_level=3)
            def func():
                qml.from_qiskit(circ)(wires=...)
                return(qml.expval(qml.PauliZ(0)))
            node = qml.QNode(func, dev)
            node.circuit.operations_in_order"""
        control0 , control1 , control2 = wires[:-1]
        target = wires[-1]
        qml.PhaseShift(0.39269908169872414, wires=[control2])
        qml.Hadamard(wires=[target])
        qml.CNOT(wires=[control2, target])
        qml.PhaseShift(-0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control2, target])
        qml.CNOT(wires=[control2, control1])
        qml.PhaseShift(-0.39269908169872414, wires=[control1])
        qml.PhaseShift(0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.PhaseShift(0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.CNOT(wires=[control2, control1])
        qml.PhaseShift(0.39269908169872414, wires=[control1])
        qml.PhaseShift(-0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.PhaseShift(-0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control1, target])
        qml.CNOT(wires=[control1, control0])
        qml.PhaseShift(-0.39269908169872414, wires=[control0])
        qml.PhaseShift(0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control2, control0])
        qml.PhaseShift(0.39269908169872414, wires=[control0])
        qml.PhaseShift(-0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(-0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control1, control0])
        qml.PhaseShift(-0.39269908169872414, wires=[control0])
        qml.PhaseShift(0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.CNOT(wires=[control2, control0])
        qml.PhaseShift(0.39269908169872414, wires=[control0])
        qml.PhaseShift(-0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(-0.39269908169872414, wires=[target])
        qml.CNOT(wires=[control0, target])
        qml.PhaseShift(0.39269908169872414, wires=[target])
        qml.Hadamard(wires=[target])

    # Qiskit transpile not compatible with newer PennyLane version so we've done this step manually above.
    '''def mcx(self, n, gate_choice="no_phase"):
        """ Define multi control x gates by using qiskit and transpiling.
            Different sets of basis gates available: "no_phase" or "ibmq"
        """
        # print(n)
        basis_gates = dict()
        basis_gates["ibmq"] = ["cx", "id", "rz", "sx", "x"]
        basis_gates["no_phase"] = ["h", "u1", "id, " "crx", "crz", "rx", "rz", "cx", "x", "z", "ccx"]

        qc = qis.QuantumCircuit(n + 1)
        qc.mcx(list(range(n)), n, mode="noancilla")
        transpiled = qis.transpile(qc, basis_gates=basis_gates[gate_choice], optimization_level=3)
        transpiled = qml.from_qiskit(transpiled)
        return transpiled'''

    def n_CNot(self, log_expr, target_wire):
        """ Add a PauliX on the target wire, controlled on everything in log_expr. """
        controls = []
        for x in log_expr.xs:
            controls.append(self.wire(x))
            if not x.equivalent(x.top):
                qml.PauliX(wires=self.wire(x))

        n_CNot_circuit = self.mcx_circuits[len(log_expr.xs)]
        n_CNot_circuit(wires=controls + [target_wire])

        for x in log_expr.xs:
            if not x.equivalent(x.top):
                qml.PauliX(wires=self.wire(x))

    def dnf_to_gates(self, log_expr, target):
        """ Turn a logic expression of what to control a Not gate on into actual gates for the circuit.
            Assumes the log_expr to be in dnf form, so the outermost operation is an Or.
            Works in a recursive fashion:
                1. Add Not controlled on the first argument to circuit.
                2. Replace this argument with its negation in the log_expr and simplify.
                3. Call on dnf_to_gates with the new log_expr.
        """
        if log_expr.depth > 0:
            if log_expr.NAME == "Or":
                inputs = log_expr.xs
                current = inputs[0]
                if current.depth < 1:
                    self.one_CNot(current, target)
                elif current.NAME == "And":
                    self.n_CNot(current, target)
                else:
                    raise Exception(f"Not sure how to handle {current}")

                new_expr = And(Or(*inputs[1:]), Not(current)).to_dnf()
                self.dnf_to_gates(new_expr, target)
            elif log_expr.NAME == "And":
                self.n_CNot(log_expr, target)

        else:
            self.one_CNot(log_expr, target)

    def one_CNot(self, log_expr, target_wire):

        if log_expr.is_one():
            qml.PauliX(wires=target_wire)
        elif log_expr.is_zero():
            return
        else:
            if not log_expr.equivalent(log_expr.top):
                qml.PauliX(wires=self.wire(log_expr))
            qml.CNOT(wires=[self.wire(log_expr), target_wire])
            if not log_expr.equivalent(log_expr.top):
                qml.PauliX(wires=self.wire(log_expr))

    def bin_list(self, number):
        return list(np.binary_repr(number, width=self.n))

    def circuit(self, image=np.array([[]])):
        """ Encode a 2^n x 2^n picture with 2^q values in q + 2n qubits.
            Use the espresso algorithm to optimize the logic expressions that are derived from
            the combinations of Toffoli-Gates (basically apply PauliX to the target, if XOr(inputs) == True).
            This method is described in the Zhang et al paper.
        """

        image = (image * (2 ** self.q - 1)).clone().detach().round()
        # we assume the qubits start out in |0>.
        for wire in range(2 * self.n):
            qml.Hadamard(wires=self.q + wire)

        coordinates = np.array(list(map(self.bin_list, range(2 ** self.n))))
        binaries = [np.binary_repr(int(i), width=self.q) for i in image.flatten()]

        for qubit in range(self.q):
            toffolis = [i[qubit] for i in binaries]

            controls = [And(*np.where(coordinates[i // 2 ** self.n], self.qubits_ev[-self.n:],
                                      self.not_qubits_ev[-self.n:]).tolist(),
                            *np.where(coordinates[i % (2 ** self.n)], self.qubits_ev[-(2 * self.n):-self.n],
                                      self.not_qubits_ev[-(2 * self.n):-self.n]).tolist())
                        for i in range(2 ** (2 * self.n)) if toffolis[i] == "1"]

            if len(controls) > 0:
                simpl_controls = espresso_exprs(Xor(*controls).to_dnf())

                self.dnf_to_gates(simpl_controls[0], qubit)


class NEQR_unoptimised(NEQR):
    """ NEQR as above without the Espresso Optimization procedure. """
    def __init__(self, q=8, filter_length=2):
        super().__init__(q, filter_length)
        self.qubit_list = list(range(self.required_qubits))
        self.name = "NEQR_unoptimised"

    def bin_list(self, number):
        # return list(np.binary_repr(number, width=self.n))
        return [int(i) for i in np.binary_repr(number, width=self.n)]

    def circuit(self, image=np.array([[]])):

        image = (image * (2 ** self.q - 1)).clone().detach().round()
        # we assume the qubits start out in |0>.
        for wire in range(2 * self.n):
            qml.Hadamard(wires=self.q + wire)

        coordinates = np.array(list(map(self.bin_list, range(2 ** self.n))))
        binaries = [np.binary_repr(int(i), width=self.q) for i in image.flatten()]
        # print(binaries)
        for x, x_coords in enumerate(coordinates):
            for y, y_coords in enumerate(coordinates):
                # import pdb; pdb.set_trace()
                full_coords = np.concatenate((x_coords, y_coords))
                for idx, val in enumerate(full_coords):
                    if val == 0:
                        qml.PauliX(wires=self.qubit_list[-2 * self.n + idx])
                binary = binaries[x + y * 2 ** self.n]

                for idx, bit_value in enumerate(binary):
                    if bit_value == "1":
                        # print(x+ y*2**self.n, idx)
                        # import pdb; pdb.set_trace()
                        self.mcx_circuits[2 * self.n](wires=self.qubit_list[-2 * self.n:] + [idx])
                for idx, val in enumerate(full_coords):
                    if val == 0:
                        qml.PauliX(wires=self.qubit_list[-2 * self.n + idx])
