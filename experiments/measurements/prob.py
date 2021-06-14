from circuitcomponents import CircuitComponents
import pennylane as qml


class Prob_measurement(CircuitComponents):
    """ qml.prob
        Measures the likelihood of the different basis states.
        Returns 2**(#qubits) values.
    """

    def __init__(self, qubits, wires_to_act_on=None):
        self.qubits = qubits
        self.name = "Prob_measurement"
        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.num_out = 2 ** len(self.wires_to_act_on)

        self.required_qubits = max(self.wires_to_act_on) + 1

    def circuit(self):
        return qml.probs(wires=self.wires_to_act_on)
