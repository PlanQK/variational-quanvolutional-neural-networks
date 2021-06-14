import logging
import os
import numpy as np


def setup_custom_logger(name, hyperparams_dicts=[], file_logger=False, log_dir=None, logging_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(ch)
    if file_logger:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        fh = logging.FileHandler(log_dir + "/log.txt")
        fh.setLevel(logging_level)
        logger.addHandler(fh)
    if hyperparams_dicts:
        logger.info('Initialize application with hyperparameters:')
    for dict in hyperparams_dicts:
        logger.info("%s", dict)
    return logger


def setup_csv_logger(file_logger=False, log_dir=None, level=logging.DEBUG):
    """ Depreceated """
    logger = logging.getLogger("csv")
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ch.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(ch)
    if file_logger:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        fh = logging.FileHandler(log_dir + "/results.csv")
        fh.setLevel(level)
        logger.addHandler(fh)
    logger.info('Phase,Phase_Counter,Accuracy,Loss,Execution_Time_per_Step')
    return logger


class LoggingUtils:
    pass


def prepare_hparams_for_tensorboard(hyper_params, circuit0_params={}):
    """ Combine the three hyperparameter dictionaries into one, so that they can be saved in the tensorboard.
    """
    hparams_for_writer = {k: convert_types_for_tensorboard(v) for k, v in hyper_params.items()}
    hparams_for_writer.update({f'{k}@c': convert_types_for_tensorboard(v) for k, v in circuit0_params.items()})
    return hparams_for_writer


def convert_types_for_tensorboard(value):
    """ Tensorboard can only take variables that are str, bool or numerical. Here we catch a subset of the wrong
    types. """
    v = value
    if type(v) in [type(None), dict]:
        v = str(v)
    return v


def prepare_metrics_for_tensorboard(results):
    """ Definetly not final version, these are the wrong metrics and just for testing."""
    metrics_for_tensorboard = {
        "metric/ExecutionTimePerStep": np.mean(results["train/Execution Times"]),
        "metric/MaxMeanEpochAccuracy/train": max(results["train/Mean Epoch Accuracy"]),
        "metric/MinEpochLoss/train": min(results["train/Mean Epoch Loss"]),
        "metric/MaxMeanEpochAccuracy/validation": max(results["validation/Mean Epoch Accuracy"]),
        "metric/MintMeanEpochLoss/validation": min(results["validation/Mean Epoch Loss"]),
    }
    if results["test/Mean Epoch Accuracy"]:
        metrics_for_tensorboard.update({"metric/Mean Epoch Accuracy/test": results["test/Mean Epoch Accuracy"]})

    return metrics_for_tensorboard
