import sys
from datetime import datetime
import logging
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import yaml

module_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(module_dir)

from models.qnn import initialize_QNN_model
from utils.params_utils import load_params
from learner import Learner
from utils.logging_utils import setup_custom_logger
from utils.summarywriter_alternative import SummaryWriter
from data.mnist import MNISTDataset, QuantumEncodedMNISTOnDisk
from utils.parallel import run_experiments_parallel
from utils.model_utils import get_model_and_yaml_paths


def test_models(paths, params):
    time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    runners = []
    for model_path, yaml_path in get_model_and_yaml_paths(paths, params['best_val_loss_model']):
        #yaml_path = get_closest_file(path, file_names=["run-params.yaml", "hyper-params.yaml"])
        model_name = os.path.split(model_path)[-1].split('.')[0]
        results_dir = os.path.join(os.path.split(os.path.split(model_path)[0])[0], f"test-run_date_{time}_model_{model_name}")
        runners.append(TestRunner(model_path, yaml_path, results_dir, params))

    if params['multiprocessing']:
        run_experiments_parallel(runners, params['processes'])
    else:
        for runner in runners:
            runner.run()


def load_MNIST_test_set(params):
    print(f"Creating MNIST test dataset of size {params['test_samples']} with seed {params['data_shuffle_seed']}")
    print(f"Images will have size {params['img_size']}")
    print(params)
    if not params['preencoded']:
        return MNISTDataset(train_size=0, valid_size=0, test_size=params['test_samples'],
                                seed=params["data_shuffle_seed"],
                                logger=None,
                                img_size=params["img_size"]).test_set
    else:
        return QuantumEncodedMNISTOnDisk(params, params["q-device"],
                                         train_size=params['train_samples'], valid_size=params['valid_samples'], test_size=params['test_samples'],
                                         seed=params["data_shuffle_seed"],
                                         logger=None,
                                         img_size=params["img_size"]).test_set


    #test_models(params['paths'], test_set, params)


class TestRunner:
    class TestData:
        def __init__(self, test_set):
            self.test_set = test_set

    def __init__(self, model_path, model_yaml_path, results_dir, params):
        self.model_path = model_path
        self.results_dir = results_dir
        self.params = params
        with open(model_yaml_path, 'r') as file:
            self.exp_params = yaml.safe_load(file)
        self.exp_params.update(params)
        if 'name' in self.exp_params.keys():
            logger_name = self.exp_params['name']
        else:
            logger_name = os.path.split(results_dir)[0]

        if self.exp_params['data'] == 'MNIST':
            self.test_set = load_MNIST_test_set(self.exp_params)
        else:
            raise KeyError(f"Dataset {self.exp_params['data']} not implemented yet")

        self.batch_size = params['batch_size']
        self.exp_params['batch_size'] = self.batch_size
        self.logger = setup_custom_logger(logger_name,
                                          hyperparams_dicts=[params],
                                          file_logger=True,
                                          log_dir=self.results_dir,
                                          logging_level=params['logging_level'])
        self.logger.debug(f"Model will be loaded from path {model_path}")
        self.logger.debug(f"Found corresponding experiment yaml at {model_yaml_path}")
        self.writer = None
        self.write_params()

    def run(self):
        self.logger.info("Loading model")
        model, circuit = initialize_QNN_model(self.exp_params, self.exp_params['q-device'], not self.exp_params['preencoded'])
        load_params(model, torch.optim.Adam(model.parameters()), self.model_path)

        self.writer = SummaryWriter(log_dir=self.results_dir)

        learner = Learner(TestRunner.TestData(self.test_set),
                          self.exp_params,
                          [0],
                          criterion=torch.nn.CrossEntropyLoss(),
                          optimizer=None,
                          results_dir=self.results_dir,
                          logger=self.logger,
                          writer=self.writer,
                          num_workers=0,
                          only_test=True,
                          move_axis_1_to_3=not self.exp_params['preencoded'])

        learner.test(model)

        if self.writer is not None:
            self.writer.close()

    def write_params(self):
        self.logger.debug(f"hyper parameters: {self.params}")
        with open(os.path.join(self.results_dir, "test-params.yaml"), 'w') as yaml_file:
            yaml.dump(self.params, yaml_file)


test_params = {
    # Either paths to models or directories that should contain models.
    # If a path points to a directory then either the last model or the model with the lowest validation loss is chosen.
    # Depends on whether "best_val_loss_model" is set.
    'paths': [
        'save'
    ],

    # Set to False for last trained model
    # Set to True for model with lowest validation loss.
    'best_val_loss_model': True,

    #'test_samples': 10,

    'data_shuffle_seed': 362356,
    'batch_size': 1,
    'logging_level': logging.DEBUG,

    'multiprocessing': False,
    'processes': len(os.sched_getaffinity(0)),
    
    #"q-device": "qiskit.ibmq",
    #"q-device": "qulacs.simulator",
    "q-device": "default.qubit",

    'preencoded': False,
    "run_test": True
}

if __name__ == '__main__':
    test_models(test_params['paths'], test_params)
