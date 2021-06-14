import os.path
import torch
from pennylane import numpy as np
from yaml import dump

from data.mnist import MNISTDataset, QuantumEncodedMNIST, QuantumEncodedMNISTOnDisk
from models.qnn import initialize_QNN_model
from learner import Learner
from utils.random_util import GeneratorUtils
from utils.logging_utils import prepare_hparams_for_tensorboard
from utils.logging_utils import prepare_metrics_for_tensorboard
from utils.logging_utils import setup_custom_logger
from utils.summarywriter_alternative import SummaryWriter


class ExperimentRunner:

    def __init__(self, run_params, experiment_dir, q_device, logging_level, save_preload=True, pre_encoding=True):

        self.save_preload = save_preload
        self.pre_encoding = pre_encoding
        self.experiment_dir = experiment_dir
        self.q_device = q_device
        os.makedirs(self.experiment_dir, exist_ok=True)
        torch.manual_seed(run_params['torch_seed'])
        np.random.seed(run_params['np_seed'])
        self.run_params = run_params
        self.seeds = GeneratorUtils.generate_seeds(size=self.run_params["epochs_num"])
        self.logger = setup_custom_logger(run_params['name'],
                                          hyperparams_dicts=[self.run_params],
                                          file_logger=True,
                                          log_dir=self.experiment_dir,
                                          logging_level=logging_level)
        self.writer = None
        self.learner = None
        self.experiment_result = None

        self.data = None
        self.get_data()

        self.write_params()

    def run(self):
        # Multi-threading doesn't let file writers be started before
        self.writer = SummaryWriter(log_dir=self.experiment_dir)

        model_q, circuit = initialize_QNN_model(self.run_params, self.q_device, not self.pre_encoding)

        learner = Learner(data=self.data,
                          hyperparams=self.run_params,
                          seeds=self.seeds,
                          criterion=torch.nn.CrossEntropyLoss(),
                          optimizer=torch.optim.Adam(params=model_q.parameters(), lr=self.run_params['lr']),
                          results_dir=self.experiment_dir,
                          logger=self.logger,
                          writer=self.writer,
                          loader_random_seed=self.run_params['data_shuffle_seed'],
                          move_axis_1_to_3=not self.pre_encoding)

        self.experiment_result = learner.training_experiment(model_q,
                                                             self.run_params["trainable"],
                                                             1,
                                                             1,
                                                             run_test=self.run_params["run_test"])

        self.experiment_result['name'] = self.run_params['name']

        self.publish_results()

        if self.writer is not None:
            self.writer.close()

        return self.experiment_result

    def get_data(self):
        if self.run_params['data'] == 'MNIST':
            if not self.pre_encoding:
                self.data = MNISTDataset(self.run_params['train_samples'],
                                         self.run_params['valid_samples'], self.run_params['test_samples'],
                                         self.run_params['data_shuffle_seed'], logger=self.logger,
                                         img_size=self.run_params['img_size'])
            else:
                self.data = QuantumEncodedMNISTOnDisk(self.run_params, self.q_device, self.run_params['train_samples'],
                                                      self.run_params['valid_samples'], self.run_params['test_samples'],
                                                      self.run_params['data_shuffle_seed'],
                                                      logger=self.logger, img_size=self.run_params['img_size'])

    def publish_results(self):

        results = {"train/Execution Times": self.experiment_result["execution_times"],
                   "train/Mean Epoch Accuracy": self.experiment_result["accuracy_viz"],
                   "train/Mean Epoch Loss": self.experiment_result["loss_viz"],
                   "validation/Mean Epoch Accuracy": self.experiment_result["accuracy_viz_validation"],
                   "validation/Mean Epoch Loss": self.experiment_result["loss_viz_validation"],
                   "test/Mean Epoch Accuracy": self.experiment_result["accuracy_test"]
                   }

        hp = prepare_hparams_for_tensorboard(self.run_params)
        metrics = prepare_metrics_for_tensorboard(results)
        if self.writer is not None:
            self.writer.add_hparams(hp, metrics)

    def write_params(self):
        self.logger.debug(f"hyper parameters: {self.run_params}")
        with open(os.path.join(self.experiment_dir, "run-params.yaml"), 'w') as yaml_file:
            dump(self.run_params, yaml_file)

    def __str__(self):
        return str(self.run_params)
