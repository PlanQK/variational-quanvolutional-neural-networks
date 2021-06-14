import unittest
import torch
import numpy as np
from runners import ExperimentRunner
from learner import Learner
from models.qnn import initialize_QNN_model
import logging
import os
from datetime import datetime
from utils.summarywriter_alternative import SummaryWriter

hyper_params = {
    "out_channels": 4,  # depends on the encoding
    "circuit_layers": 1,
    "n_rotations": 4,
    "filter_length": 2,
    "stride": 1,
    "out_features": 10,
    "batch_size": 2,
    "epochs_num": 2,
    "steps_in_epoch": 2,
    "val_data_size": 2,
    "train_split_percent": 0.8,
    "run_test": True,
    "test_samples": 10,
    "data": 'MNIST',
    "img_size": 14,  # use 28x28 images; change for resize
    "encoder": "Threshold_Encoder",
    "encoder_args": {},
    "data_shuffle_seed": 362356,
    "weights_seed": 11111,
    "torch_seed": 10,
    "np_seed": 10,
    "lr": 0.01,
    "logs_path": './save/',
    "calculation": "RandomLayer",
    "calculation_seed": 10,
    "calculation_args": {},
    "measurement": "UniformGateMeasurements",
    "measurement_args": {},
    "trainable": True
}


class TestDatasetIntegration(unittest.TestCase):
    def setUp(self) -> None:
        """ Called once before every test."""
        root_dir = str('./save/unittests' + '_date_' +
                       datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        self.runners = []
        for i in range(2):
            exp_dir = os.path.join(root_dir, str(i))
            hyper_params["name"] = str(i)
            self.runners.append(ExperimentRunner(hyper_params, exp_dir, logging.INFO,
                                                 multiprocesing=False))
            writer = SummaryWriter(log_dir=exp_dir)
            logger = logging.getLogger("root")
            self.runners[i].model_q, circuit = initialize_QNN_model(hyper_params)

            self.runners[i].learner = Learner(data=self.runners[i].data,
                                              hyperparams=self.runners[i].run_params,
                                              seeds=self.runners[i].seeds,
                                              criterion=torch.nn.CrossEntropyLoss(),
                                              optimizer=torch.optim.Adam(params=self.runners[i].model_q.parameters(),
                                                                         lr=self.runners[i].run_params['lr']),
                                              results_dir=self.runners[i].experiment_dir,
                                              logger=logger,
                                              writer=writer,
                                              loader_random_seed=self.runners[i].run_params['data_shuffle_seed'])

        self.result0 = self.runners[0].learner.training_experiment(self.runners[0].model_q, False, 1, 1,
                                                                   is_unittest_case=True)
        self.result1 = self.runners[1].learner.training_experiment(self.runners[1].model_q, True, 1, 1,
                                                                   is_unittest_case=True)

    def test_train_data_is_sampled_in_the_same_order(self):
        np.testing.assert_array_equal(self.result0["train_data"],
                                      self.result1["train_data"])

    def test_val_data_is_sampled_in_the_same_order(self):
        np.testing.assert_array_equal(self.result0["val_data"],
                                      self.result1["val_data"])


if __name__ == "__main__":
    logging.disable(2)
    unittest.main()
