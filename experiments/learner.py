import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from utils.calculation_utils import calculate_mean_of_metric, calculate_execution_time, predict
from utils.params_utils import save_params
from utils.random_util import set_seed


class Learner:

    def __init__(self, data, hyperparams, seeds, criterion, optimizer, results_dir, logger, writer=None,
                 loader_random_seed=0,
                 num_workers=0,
                 only_test=False,
                 move_axis_1_to_3=True):
        self.data = data
        self.seeds = seeds
        self.batch_size = hyperparams['batch_size']

        if not only_test:
            self.epoch = hyperparams["epochs_num"]
            self.val_data_size = hyperparams["val_data_size"]
            self.steps_in_epoch = hyperparams["steps_in_epoch"]
            self.optimizer = optimizer
            self.model_checkpoint_dir = os.path.join(results_dir, 'model_checkpoints')
            os.makedirs(self.model_checkpoint_dir, exist_ok=True)

        self.logger = logger
        self.writer = writer
        self.criterion = criterion
        self.results_dir = results_dir
        self.move_axis = move_axis_1_to_3
        if not only_test:
            self.train_loader = DataLoader(self.data.train_set, batch_size=self.batch_size, num_workers=num_workers,
                                           worker_init_fn=lambda worker_id: np.random.seed(
                                               loader_random_seed + worker_id),
                                           shuffle=True)
            if self.data.valid_set is not None:
                self.valid_loader = DataLoader(self.data.valid_set, batch_size=self.batch_size, num_workers=num_workers,
                                               worker_init_fn=lambda worker_id: np.random.seed(
                                                   loader_random_seed + worker_id),
                                               shuffle=True)

        self.test_loader = DataLoader(self.data.test_set, batch_size=self.batch_size, num_workers=num_workers,
                                      shuffle=False)
        self.img_size = hyperparams["img_size"]

    def train(self, model, trainable, is_unittest_case=False):
        accuracy_viz, accuracy_viz_validation, execution_times = [], [], []
        loss_viz, loss_viz_validation = [], []
        training_data = []

        for epoch in range(1, self.epoch + 1):
            model.train()
            self.logger.info("Start training epoch %d with seed %d ", epoch, self.seeds[epoch - 1])
            accuracy, losses = [], []
            step = 0
            set_seed(self.seeds[epoch - 1])
            start = time.time()
            for x, y in self.train_loader:
                if is_unittest_case:
                    training_data.append(y.numpy())
                y_pred = predict(model, x, self.move_axis)
                if step == 0:
                    # weights = model.qlayer_1.torch_qlayer.weights
                    self.logger.debug(f"Quantum circuit at the beginning of epoch {epoch}")
                    model.draw_qlayer_circuit()

                # error, gradients and optimization
                loss = self.criterion(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = accuracy_score(y, y_pred.argmax(-1).numpy())
                accuracy.append(acc)
                losses.append(loss.item())

                if (step > 0) and (step % 5 == 0):
                    if trainable:
                        self.logger.debug("Training step %d. Output current quantum circuit weights.", step)
                        self.logger.debug(model.qlayer_1.torch_qlayer.weights)

                # early break
                if step > 0 and step % (self.steps_in_epoch - 1) == 0:
                    break
                step += 1

            save_params(model, self.optimizer, epoch, self.model_checkpoint_dir)

            mean_epoch_accuracy = calculate_mean_of_metric(accuracy)
            mean_epoch_loss = calculate_mean_of_metric(losses)
            execution_time_step = (time.time() - start) / len(accuracy)

            self.logger.info("Epoch %d / %d . Accuracy: %f. Loss: %f. Time: %f",
                             epoch, self.epoch, mean_epoch_accuracy, mean_epoch_loss, execution_time_step)
            self.write(model, execution_time_step, mean_epoch_loss, mean_epoch_accuracy, epoch)
            accuracy_viz.append(mean_epoch_accuracy)
            loss_viz.append(mean_epoch_loss)
            execution_times.append(execution_time_step)

            if self.data.valid_set is not None:
                val_acc, val_losses, val_data = self.validate(model, epoch, is_unittest_case)
                self.logger.info("Val Accuracy: %f, Val Loss: %f ", val_acc, val_losses)
                accuracy_viz_validation.append(val_acc)
                loss_viz_validation.append(val_losses)
            else:
                val_acc = val_losses = val_data = None

            record = {
                "epoch": [epoch],
                "train_loss": [mean_epoch_loss],
                "train_acc": [mean_epoch_accuracy],
                "train_time_per_step": [execution_time_step],
                "val_loss": [val_losses],
                "val_acc": [val_acc]
            }
            self.save_results(record, "train_result.csv")

        if self.writer:
            self.writer.flush()
        return accuracy_viz, loss_viz, accuracy_viz_validation, loss_viz_validation, training_data, val_data, execution_times

    def write(self, model, execution_time_step, mean_epoch_loss, mean_epoch_accuracy, epoch):
        if self.writer:
            self.writer.add_scalar("Execution Time/train", execution_time_step, epoch)
            self.writer.add_scalar("Mean Epoch Loss/train", mean_epoch_loss, epoch)
            self.writer.add_scalar("Mean Epoch Accuracy/train", mean_epoch_accuracy, epoch)
            for name, weight in model.named_parameters():
                self.writer.add_histogram(name, weight, epoch)
                if weight.requires_grad:
                    self.writer.add_histogram(f"{name}.grad", weight.grad, epoch)

    def save_results(self, record_dict, file_name):
        results_file = os.path.join(self.results_dir, file_name)
        df = pd.DataFrame.from_dict(record_dict)
        if os.path.isfile(results_file):
            df_prev = pd.read_csv(results_file)
            df = df_prev.append(df, ignore_index=True)

        df.to_csv(results_file, index=False)

    def validate(self, model, epoch, is_unittest_case=False):
        self.logger.info("Validate after epoch %d with seed %d", epoch, self.seeds[epoch - 1])
        accuracy, losses = [], []
        validation_data = []  # for tests
        model.eval()
        step = 0
        set_seed(self.seeds[epoch - 1])
        for x, y in self.valid_loader:
            if is_unittest_case:
                validation_data.append(y.numpy())
            with torch.no_grad():
                y_pred = predict(model, x, self.move_axis)
                loss = self.criterion(y_pred, y)
                acc_val = accuracy_score(y, y_pred.argmax(-1).numpy())
                accuracy.append(acc_val)
                losses.append(loss.item())

                # early break
                if step > 0 and step % (self.val_data_size - 1) == 0:
                    break
                step += 1
        if self.writer:
            self.writer.add_scalar("Mean Epoch Loss/validation", calculate_mean_of_metric(losses), epoch)
            self.writer.add_scalar("Mean Epoch Accuracy/validation", calculate_mean_of_metric(accuracy), epoch)

            self.writer.flush()
        return calculate_mean_of_metric(accuracy), calculate_mean_of_metric(losses), validation_data

    def training_experiment(self, model, trainable, layers_num, number_of_filters, run_test=False,
                            is_unittest_case=False):
        """Train the given model
        Parameters
        ----------
        model
            QNN Model
        trainable : bool True if trainable QNN
        layers_num : int Number of quanvolution layers
        number_of_filters: int number of quanvolution filters
        run_test: bool True if run tests after the experiment
        is_unittest_case: bool True if the function is called for a unit test case
        """
        experiment_result = dict()

        start = time.time()
        if trainable:
            self.logger.info("Starting Trainable Experiment with %d layers and %d filters", layers_num,
                             number_of_filters)
        else:
            self.logger.info("Starting Untrainable Experiment with %d layers and %d filters", layers_num,
                             number_of_filters)
        for name, param in model.named_parameters():
            if name == "qlayer_1.torch_qlayer.weights" and not trainable:
                param.requires_grad = False
            else:
                param.requires_grad = True

        train_output = self.train(model, trainable, is_unittest_case)
        hours, minutes, seconds = calculate_execution_time(start, end=time.time())
        self.logger.info("Training Execution time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))
        experiment_result.update(
            zip(["accuracy_viz", "loss_viz", "accuracy_viz_validation", "loss_viz_validation", "train_data",
                 "val_data", "execution_times"],
                train_output))

        # Testing
        accuracy_test, loss_test = None, None
        if run_test:
            accuracy_test, loss_test = self.test(model)
        else:
            self.logger.info("No testing conducted")

        experiment_result["accuracy_test"], experiment_result["loss_test"] = accuracy_test, loss_test
        return experiment_result

    def test(self, model):

        self.logger.info("Conducting Test")
        accuracy = []
        losses = []
        set_seed(self.seeds[0])
        model.eval()
        for x, y in self.test_loader:
            with torch.no_grad():
                y_pred = predict(model, x, self.move_axis)
                loss = self.criterion(y_pred, y)
                acc_test = accuracy_score(y, y_pred.argmax(-1).numpy())
                accuracy.append(acc_test)
                losses.append(loss.item())

        loss_test = calculate_mean_of_metric(losses)
        accuracy_test = calculate_mean_of_metric(accuracy)
        record = {
            "test_loss": [loss_test],
            "test_acc": [accuracy_test]
        }
        self.save_results(record, "test_result.csv")

        if self.writer:
            self.writer.add_scalar("Mean Epoch Loss/test", calculate_mean_of_metric(losses), None)
            self.writer.add_scalar("Mean Epoch Accuracy/test", calculate_mean_of_metric(accuracy), None)
            self.writer.flush()

        self.logger.info("Test Accuracy: %f, Test Loss: %f ", accuracy_test, loss_test)

        return loss_test, accuracy_test
