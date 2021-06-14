from circuitcomponents import CircuitComponents
import pennylane as qml


class RandomLayer(CircuitComponents):
    """ Layer of random quantum computations. Directly from qml.templates.RandomLayers."""

    def __init__(self, qubits, seed=0, wires_to_act_on=None):
        super().__init__()
        self.seed = seed
        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Randomlayer"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        qml.templates.RandomLayers(weights, wires=self.wires_to_act_on, seed=self.seed, ratio_imprim=0.4)
