import abc
import logging
import os
import urllib
from collections import defaultdict
from threading import Thread

from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import math
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from urllib.error import HTTPError
from pathos.pools import ProcessPool

from .load_individual_tensors import LoadIndividualTensorsSorted
from .preload_dataset import PreLoadIntoMemory, PreloadFromFile, SetManually, PreloadXFromFileYManually
from utils.calculation_utils import predict
from utils.circuitcomponents_utils import generate_status_encoding_circuit, get_wires_number
from models.quonv_layer import QuonvLayer, ExtractStatesQuonvLayer


class ExperimentDataset:

    def __init__(self, seed=0, path='./data', save_preload=True, logger=None, writer=None):
        self.save_preload = save_preload
        self.path = path
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        self.writer = writer
        self.set_header()
        self.samples_test = 0
        self.split_rate = 0
        self.seed = seed

        self.save_dir = os.path.join(self.path, f"preload_tensors_saved_img_size-{self.img_size}")
        self.set_save_paths()

        self.train_set = self.valid_set = self.test_set = None

    def set_save_paths(self):
        self.train_x_path = os.path.join(self.save_dir, "train_x.pt")
        self.train_y_path = os.path.join(self.save_dir, "train_y.pt")
        self.test_x_path = os.path.join(self.save_dir, "test_x.pt")
        self.test_y_path = os.path.join(self.save_dir, "test_y.pt")

    @abc.abstractmethod
    def get_subsampled_test_dict(self):
        return

    @abc.abstractmethod
    def get_split_train_valid_dict(self):
        return

    @abc.abstractmethod
    def get_preloaded_train(self):
        return

    @abc.abstractmethod
    def get_preloaded_test(self):
        return

    def get_subsampled_test_key(self):
        return str(self.seed) + '-' + str(self.samples_test)

    def get_split_train_valid_key(self):
        return str(self.seed) + '-' + str(self.split_rate)

    def get_subsampled_test_set(self):
        key = self.get_subsampled_test_key()
        return self.get_subsampled_test_dict()[key]

    def get_split_train_valid_set(self):
        key = self.get_split_train_valid_key()
        return self.get_split_train_valid_dict()[key]

    def add_subsampled_test_set(self, test_set):
        subsampled_tests = self.get_subsampled_test_dict()
        subsampled_tests[self.get_subsampled_test_key()] = test_set

    def add_split_train_valid(self, train_set, valid_set):
        train_valid_splits = self.get_split_train_valid_dict()
        train_valid_splits[self.get_split_train_valid_key()] = (train_set, valid_set)

    """def set_valid_set_empty(self):
        self.valid_set = SetManually(torch.tensor([]), torch.tensor([]))"""

    def split_valid_set_off_preload_train(self, split_rate) -> (Subset, Subset):
        self.split_rate = split_rate
        if split_rate < 0:
            self.split_rate = 0
            self.train_set = self.get_preloaded_train()
            self.valid_set = None
        num_data = len(self.get_preloaded_train())
        split = math.floor(split_rate * num_data)
        self.logger.info(
            f"Balanced splitting with seed {self.seed}:\n\t\t\t\tTraining set: {num_data - split} samples\n\t\t\t\tValidation set: {split} samples")

        train_valid_set = self.get_split_train_valid_set()

        if train_valid_set is None:
            self.train_set, self.valid_set = self.subsample_balanced(self.get_preloaded_train(), split)
            self.add_split_train_valid(self.train_set, self.valid_set)
        else:
            self.train_set, self.valid_set = train_valid_set

    def subsample_preload_test_set(self, samples_test):
        self.samples_test = samples_test
        if samples_test <= 0:
            self.samples_test = 0
            self.test_set = self.get_preloaded_test()
            return

        self.logger.info(
            f"Balanced subsampling of test dataset. {self.samples_test} data points to be selected with seed {self.seed}.")
        self.test_set = self.get_subsampled_test_set()

        if self.test_set is None:
            _, test_set = self.subsample_balanced(self.get_preloaded_test(), self.samples_test)
            self.add_subsampled_test_set(test_set)
            self.test_set = test_set

    def subsample_balanced(self, dataset, number_of_samples):
        labels = np.empty((len(dataset)))
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels[i] = label

        sss = StratifiedShuffleSplit(1, test_size=number_of_samples, random_state=self.seed)

        for train_index, test_index in sss.split(np.arange(len(dataset)), labels):
            indices_train = train_index
            indices_test = test_index

        return Subset(dataset, indices_train), Subset(dataset, indices_test)

    def save_data_tensors_to_disk(self, preloaded_train_set, preloaded_test_set):
        self.logger.info(f"Saving data tensors to disk at {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(preloaded_train_set.x, self.train_x_path)
        torch.save(preloaded_train_set.y, self.train_y_path)
        torch.save(preloaded_test_set.x, self.test_x_path)
        torch.save(preloaded_test_set.y, self.test_y_path)

    def set_header(self) -> None:
        '''Set header for the data download (otherwise 403 Error)'''
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)


class MNISTDataset(ExperimentDataset):
    train_preloaded = None
    test_preloaded = None
    split_train_valid = defaultdict(lambda: None)
    subsampled_test = defaultdict(lambda: None)

    def __init__(self, train_size=3000, valid_size=300, test_size=300, seed=None, normalize=False, path='./mnist',
                 save_preload=True, logger=None, writer=None,
                 img_size=28):

        self.normalize = normalize
        self.img_size = img_size

        super().__init__(seed, path, save_preload, logger, writer)
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.preload_MNIST()
        self.set_MNIST_data_sets()

        self.write_data_to_tensorboard()

    def get_subsampled_test_dict(self):
        return MNISTDataset.subsampled_test

    def get_split_train_valid_dict(self):
        return MNISTDataset.split_train_valid

    def get_preloaded_test(self):
        return MNISTDataset.test_preloaded

    def get_preloaded_train(self):
        return MNISTDataset.train_preloaded

    def set_MNIST_data_sets(self):
        if self.train_size == 0:
            self.train_set = MNISTDataset.train_preloaded
        if self.valid_size == 0:
            self.valid_set = None
        if self.test_size == 0:
            self.test_set = MNISTDataset.test_preloaded

        train_valid_size = self.train_size + self.valid_size
        if train_valid_size != 0:
            _, self.train_set = self.subsample_balanced(MNISTDataset.train_preloaded, train_valid_size)
            if self.valid_size != 0:
                self.train_set, self.valid_set = self.subsample_balanced(self.train_set, self.valid_size)

        if self.test_size != 0:
            _, self.test_set = self.subsample_balanced(MNISTDataset.train_preloaded, self.test_size)

    def load_mnist_from_pytorch(self, normalize, path) -> (MNIST, MNIST):
        transforms_list = [transforms.ToTensor(), transforms.Resize(self.img_size)]
        if normalize:
            transforms_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        try:
            train_set = MNIST(root=path, train=True, download=True, transform=transforms.Compose(transforms_list))
            test_set = MNIST(root=path, train=False, download=True, transform=transforms.Compose(transforms_list))
            return train_set, test_set

        except HTTPError as err:
            if err.code == 503:
                print("Data Service Unavailable. Debug message: %s", err)
            else:
                print("An error has been occurred. Reason %s ", err)

    def preload_MNIST(self):
        if MNISTDataset.train_preloaded is None or MNISTDataset.test_preloaded is None:
            if self.save_preload and os.path.exists(self.train_x_path) and os.path.exists(self.test_x_path):
                self.logger.info(f"Loading MNIST tensors from {self.save_dir}")
                MNISTDataset.train_preloaded = PreloadFromFile(self.train_x_path, self.train_y_path)
                MNISTDataset.test_preloaded = PreloadFromFile(self.test_x_path, self.test_y_path)
            else:
                self.logger.info("Loading MNIST dataset from the PyTorch storage")
                train_set, test_set = self.load_mnist_from_pytorch(self.normalize, self.path)
                self.logger.info("Loading MNIST dataset into memory")
                MNISTDataset.train_preloaded = PreLoadIntoMemory(train_set)
                MNISTDataset.test_preloaded = PreLoadIntoMemory(test_set)

                if self.save_preload:
                    self.save_data_tensors_to_disk(MNISTDataset.train_preloaded, MNISTDataset.test_preloaded)

    def write_data_to_tensorboard(self):
        if self.writer is not None:
            dataiter = iter(MNISTDataset.train_preloaded)
            images, labels = dataiter.next()
            img_grid = torchvision.utils.make_grid(images)
            self.writer.add_image('digits_mnist_images', img_grid)
            self.writer.flush()


class QuantumEncodedMNIST(MNISTDataset):
    encoded_train_preloaded = defaultdict(lambda: None)
    encoded_test_preloaded = defaultdict(lambda: None)
    encoded_split_train_valid = defaultdict(lambda: None)
    encoded_subsampled_test = defaultdict(lambda: None)

    def __init__(self, params, q_device, train_size=3000, valid_size=300, test_size=300, seed=None, normalize=False,
                 path='./mnist-encoded', mnist_path='./mnist',
                 save_preload=True, logger=None,
                 writer=None, img_size=28, parallelization=1):
        self.params = params
        self.q_device = q_device

        self.parallelization = parallelization

        super().__init__(0, 0, 0, seed, normalize, mnist_path, save_preload, logger, writer,
                         img_size)
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.path = path
        self.mnist_path = mnist_path
        self.mnist_dir = os.path.join(self.mnist_path, f"preload_tensors_saved_img_size-{self.img_size}")
        self.save_dir = os.path.join(self.path, f"{self.get_encoded_preloaded_key()}")
        self.set_save_paths()
        self.preload_encoded_MNIST()
        self.train_set = QuantumEncodedMNIST.encoded_train_preloaded[self.get_encoded_preloaded_key()]
        self.test_set = QuantumEncodedMNIST.encoded_test_preloaded[self.get_encoded_preloaded_key()]

    def get_encoded_preloaded_key(self):
        try:
            return f"{self.params['encoder']}-{self.params['stride']}-{self.params['filter_length']}-{self.img_size}"
        except KeyError:
            raise ValueError("Parameter 'stride' and/or 'filter_length' not found")

    def get_subsampled_test_key(self):
        return str(self.seed) + '-' + str(self.samples_test) + self.get_encoded_preloaded_key()

    def get_split_train_valid_key(self):
        return str(self.seed) + '-' + str(self.split_rate) + self.get_encoded_preloaded_key()

    def get_subsampled_test_dict(self):
        return QuantumEncodedMNIST.encoded_subsampled_test

    def get_split_train_valid_dict(self):
        return QuantumEncodedMNIST.encoded_split_train_valid

    def get_preloaded_test(self):
        return QuantumEncodedMNIST.encoded_test_preloaded[self.get_encoded_preloaded_key()]

    def get_preloaded_train(self):
        return QuantumEncodedMNIST.encoded_train_preloaded[self.get_encoded_preloaded_key()]

    def preload_encoded_MNIST(self):

        if self.get_preloaded_train() is None or self.get_preloaded_test() is None:
            if self.save_preload and os.path.exists(self.train_x_path) and os.path.exists(self.test_x_path):
                self.logger.info(f"Loading {self.params['encoder']} encoded MNIST tensors from {self.save_dir}")
                QuantumEncodedMNIST.encoded_train_preloaded[
                    self.get_encoded_preloaded_key()] = PreloadXFromFileYManually(self.train_x_path,
                                                                                  MNISTDataset.train_preloaded.y)
                QuantumEncodedMNIST.encoded_test_preloaded[
                    self.get_encoded_preloaded_key()] = PreloadXFromFileYManually(self.test_x_path,
                                                                                  MNISTDataset.test_preloaded.y)

            else:

                circuit = generate_status_encoding_circuit(self.params)
                qonv = ExtractStatesQuonvLayer(weights=None,
                                               stride=self.params['stride'],
                                               device=self.q_device,
                                               wires=get_wires_number(self.params),
                                               filter_size=self.params['filter_length'],
                                               out_channels=self.params['out_channels'],
                                               circuit=circuit,
                                               dtype=torch.complex64)
                qonv.eval()
                with torch.no_grad():
                    self.logger.info(f"Encoding MNIST train set with {self.params['encoder']}")
                    preencoded_train = self.get_preencoded_tensor(qonv, MNISTDataset.train_preloaded)
                    QuantumEncodedMNIST.encoded_train_preloaded[self.get_encoded_preloaded_key()] = SetManually(
                        preencoded_train, MNISTDataset.train_preloaded.y)
                    self.logger.info(f"Encoding MNIST test set with {self.params['encoder']}")
                    preencoded_test = self.get_preencoded_tensor(qonv, MNISTDataset.test_preloaded)
                    QuantumEncodedMNIST.encoded_test_preloaded[self.get_encoded_preloaded_key()] = SetManually(
                        preencoded_test,
                        MNISTDataset.test_preloaded.y)
                    if self.save_preload:
                        self.save_encoded_data_tensors_to_disk(
                            QuantumEncodedMNIST.encoded_train_preloaded[self.get_encoded_preloaded_key()],
                            QuantumEncodedMNIST.encoded_test_preloaded[self.get_encoded_preloaded_key()])

    def save_encoded_data_tensors_to_disk(self, preloaded_train_set, preloaded_test_set):
        self.logger.info(f"Saving data tensors to disk at {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(preloaded_train_set.x, self.train_x_path)
        torch.save(preloaded_test_set.x, self.test_x_path)

    def get_preencoded_tensor(self, qonv, dataset):
        x_tensor = None
        # data_len = len(dataset)
        data_len = 1

        for i in tqdm(range(data_len)):

            x, _ = dataset[i]
            qonv_states = predict(qonv, x[None], True)[0]

            if i == 0:
                x_tensor = torch.empty((data_len,) + tuple(qonv_states.shape), dtype=torch.complex64)

            x_tensor[i] = qonv_states
        return x_tensor


class QuantumEncodedMNISTOnDisk(QuantumEncodedMNIST):

    def __init__(self, params, q_device, train_size=3000, valid_size=300, test_size=300, seed=None, normalize=False,
                 path='./mnist-encoded', mnist_path='./mnist',
                 save_preload=True, logger=None,
                 writer=None, img_size=28, parallelization=3):

        #assert train_size > 0 and test_size > 0, "Embeddings can be huge, recommended to subsample"
        super().__init__(params, q_device, train_size=train_size, valid_size=valid_size, test_size=test_size, seed=seed,
                         normalize=normalize,
                         path=path, mnist_path=mnist_path,
                         save_preload=save_preload, logger=logger,
                         writer=writer, img_size=img_size, parallelization=parallelization)

        self.train_set = LoadIndividualTensorsSorted(self.train_val_dir)
        if self.valid_size != 0:
            self.train_set, self.valid_set = self.subsample_balanced(self.train_set, self.valid_size)

        self.test_set = LoadIndividualTensorsSorted(self.test_dir)

    def get_subsampled_test_key(self):
        return str(self.seed) + '-' + str(self.samples_test) + self.get_encoded_preloaded_key()

    def get_split_train_valid_key(self):
        return str(self.seed) + '-' + str(self.split_rate) + self.get_encoded_preloaded_key()


    def preload_encoded_MNIST(self):
        self.train_val_dir = os.path.join(self.save_dir, f"train_val-{self.seed}-{self.train_size + self.valid_size}")
        self.test_dir = os.path.join(self.save_dir, f"test-{self.seed}-{self.test_size}")

        if not os.path.exists(os.path.join(self.train_val_dir, f"{0}.pt")) or not os.path.exists(os.path.join(self.test_dir, f"{0}.pt")):
            os.makedirs(self.test_dir, exist_ok=True)
            os.makedirs(self.train_val_dir, exist_ok=True)
            to_encode_train_set = self.subsample_balanced(MNISTDataset.train_preloaded, self.train_size + self.valid_size)[1]
            to_encode_test_set = self.subsample_balanced(MNISTDataset.train_preloaded, self.test_size)[1]

            def split_indices(size, num_parts):
                part_length = math.floor(size / num_parts)
                splits = [(j*part_length, (j + 1) * part_length) for j in range(num_parts - 1)]
                splits.append(((num_parts - 1) * part_length, size))
                return splits

            if self.parallelization > 2:
                parallel_tuples = [(to_encode_train_set, *split, self.train_val_dir) for i, split in
                                   enumerate(split_indices(len(to_encode_train_set), self.parallelization - 1))]
                parallel_tuples.append((to_encode_test_set, 0, len(to_encode_test_set), self.test_dir))
                pool = ProcessPool(self.parallelization)
                pending_results = pool.amap(
                    lambda parallel_tuple: self.encode_part(*parallel_tuple), parallel_tuples)
                results = pending_results.get()
                pool.close()
                pool.join()
            else:
                circuit = generate_status_encoding_circuit(self.params)
                qonv = ExtractStatesQuonvLayer(weights=None,
                                                stride=self.params['stride'],
                                                device=self.q_device,
                                                wires=get_wires_number(self.params),
                                                filter_size=self.params['filter_length'],
                                                out_channels=self.params['out_channels'],
                                                circuit=circuit,
                                                dtype=torch.complex64)
                qonv.eval()
                with torch.no_grad():

                    self.logger.info(f"Encoding MNIST train set with {self.params['encoder']}")
                    for i, encoded_img in enumerate(self.iter_over_dataset(qonv, to_encode_train_set)):
                        torch.save(encoded_img, os.path.join(self.train_val_dir, f"{i}.pt"))

                    self.logger.info(f"Encoding MNIST test set with {self.params['encoder']}")
                    for i, encoded_img in enumerate(self.iter_over_dataset(qonv, to_encode_test_set)):
                        torch.save(encoded_img, os.path.join(self.test_dir, f"{i}.pt"))

    def encode_part(self, dataset, start, end, dir):
        print(f"{self.params['encoder']} encode", start, end)
        circuit = generate_status_encoding_circuit(self.params)
        qonv = ExtractStatesQuonvLayer(weights=None,
                                        stride=self.params['stride'],
                                        device=self.q_device,
                                        wires=get_wires_number(self.params),
                                        filter_size=self.params['filter_length'],
                                        out_channels=self.params['out_channels'],
                                        circuit=circuit,
                                        dtype=torch.complex64)
        qonv.eval()
        with torch.no_grad():
            for i, encoded_img in enumerate(self.iter_over_dataset(qonv, dataset, start, end)):
                torch.save(encoded_img, os.path.join(dir, f"{i + start}.pt"))

    def iter_over_dataset(self, qonv, dataset, start=None, end=None):
        data_len = len(dataset)
        # data_len = 2
        if start is not None and end is not None:
            indices = range(start, end)
        elif start is not None:
            indices = range(start, data_len)
        elif end is not None:
            indices = range(0, end)
        else:
            indices = range(0, data_len)
        for i in tqdm(indices):

            x, y = dataset[i]
            qonv_states = predict(qonv, x[None], True)[0]

            yield qonv_states, y


class QuantumEncodedMNISTOnDiskThread(Thread):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        Thread.__init__(self)

    def run(self):
        QuantumEncodedMNISTOnDisk(*self.args, **self.kwargs)
