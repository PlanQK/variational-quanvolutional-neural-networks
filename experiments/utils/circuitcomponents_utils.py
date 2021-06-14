from encoders.threshold_encoder import Threshold_Encoder
from encoders.frqi import FRQI_for_2x2
from encoders.frqi import FRQI_for_4x4
from encoders.neqr import NEQR
from encoders.neqr import NEQR_unoptimised
from calculations.random_layer import RandomLayer
from measurements.uniform_gate import UniformGateMeasurements

import pennylane as qml

circuit_dict = {
    "Threshold_Encoder": Threshold_Encoder,
    "FRQI_for_2x2": FRQI_for_2x2,
    "FRQI_for_4x4": FRQI_for_4x4,
    "NEQR": NEQR,
    "NEQR_unoptimised": NEQR_unoptimised,
    "RandomLayer": RandomLayer,
    "UniformGateMeasurements": UniformGateMeasurements
}


def generate_circuit(encoding, calculation, measurement=None):
    """ Combine the three steps encoding, calculation and measurement into a singular pennylane circuit.
        Allows for plug in testing of different options in our QNN.
        All inputs must be sequences of pennylane gates. Measurements can be included directly in calculation
        or explicitly in measurement.
    """
    if measurement is not None:
        def func(inputs, weights):
            encoding(inputs)
            calculation(weights)
            result = measurement()
            return result
    else:
        def func(inputs, weights):
            encoding(inputs)
            result = calculation(weights)
            return result
    return func


def generate_circuit_pre_encoded_input(calculation, q_bits, measurement=None):
    if measurement is not None:
        def calc(weights):
            calculation(weights)
            return measurement()
    else:
        calc = calculation

    def pre_encoded_calc(inputs, weights):
        qml.QubitStateVector(inputs, wires=list(range(q_bits)))
        inputs = inputs.float()
        results = calc(weights)
        return results

    return pre_encoded_calc


def generate_corresponding_circuit(hyperparams, weights_initialized=None, encoding=True):
    """ Wrap the entire circuit creation in here."""
    try:
        encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"],
                                                       **hyperparams["encoder_args"])
        calculation = circuit_dict[hyperparams["calculation"]](encoder.required_qubits,
                                                               seed=hyperparams["calculation_seed"],
                                                               **hyperparams["calculation_args"])
        measurement = circuit_dict[hyperparams["measurement"]](encoder.required_qubits,
                                                               **hyperparams["measurement_args"])
        if not hyperparams["trainable"]:
            calculation.circuit = make_untrainable(calculation.circuit, weights_initialized)
        if encoding:
            return generate_circuit(encoder.circuit, calculation.circuit, measurement.circuit)
        else:
            return generate_circuit_pre_encoded_input(calculation.circuit, encoder.required_qubits, measurement.circuit)

    except KeyError:
        raise Exception(f"Most likely a circuit specified could not be found. Available are {circuit_dict.keys()}")


def make_untrainable(circuit, weights_initialized):
    """ Render a circuit untrainable, by ignoring the passed parameters called weights, since pytorch differentiates wrt
    those only. """

    def circuit_var(weights):
        circuit(weights_initialized)

    return circuit_var


def get_wires_number(hyperparams):
    encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"],
                                                   **hyperparams["encoder_args"])
    return encoder.required_qubits


def generate_status_encoding_circuit(hyperparams):
    try:
        encoding = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"],
                                                        **hyperparams["encoder_args"])

        def get_encoded_state(inputs):
            encoding.circuit(inputs)
            return qml.state()

        return get_encoded_state
    except KeyError:
        raise Exception(f"Most likely a circuit specified could not be found. Available are {circuit_dict.keys()}")
