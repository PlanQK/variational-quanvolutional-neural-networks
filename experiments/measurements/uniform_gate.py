import pennylane as qml

from circuitcomponents import CircuitComponents


class UniformGateMeasurements(CircuitComponents):
    """ Measure PauliZ on all qubits or on the qubits specified in wires_to_act_on. """

    def __init__(self, qubits, gates=qml.PauliZ, wires_to_act_on=None, sample=False):
        self.qubits = qubits
        self.gates = gates
        self.name = "PauliZ_measurement"

        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        if sample:
            self.method = qml.sample
        else:
            self.method = qml.expval
        self.num_out = len(self.wires_to_act_on)

        self.required_qubits = max(self.wires_to_act_on) + 1

    def circuit(self):
        return [self.method(self.gates(wires=i)) for i in self.wires_to_act_on]
