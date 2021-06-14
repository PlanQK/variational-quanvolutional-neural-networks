from pennylane import numpy as np


class CalculationUtils:
    pass


def calculate_mean_of_metric(metric):
    return np.mean(metric)


def calculate_execution_time(start, end) -> (int, int, int):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def predict(model, x, move_axis_1_to_3=True):
    if move_axis_1_to_3:
        return model(x.permute(0, 2, 3, 1))
    else:
        return model(x)

