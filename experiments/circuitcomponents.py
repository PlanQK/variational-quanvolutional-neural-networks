class CircuitComponents:
    """ Parent class of the Components passed to generate_circuit. """

    def __init__(self):
        self.required_qubits = None
        self.name = None
        self.available_encoders = ["Threshold_Encoder", "NEQR", "FRQI_for_2x2"]
        self.available_calculations = ["Randomlayer", "Nothing"]
        self.available_measurements = ["Uniform_gate_measurements", "Prob_measurement"]

    def circuit(self, weights):
        pass
