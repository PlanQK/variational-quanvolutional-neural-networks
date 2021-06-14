import os, sys
from threading import Thread

module_dir = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
sys.path.append(module_dir)

from collections import defaultdict
from pathos.pools import ProcessPool
from data.mnist import QuantumEncodedMNIST, MNISTDataset, QuantumEncodedMNISTOnDisk, \
    QuantumEncodedMNISTOnDiskThread
from main import circuits


def get_param_data_key(run_params):
    return f"{run_params['encoder']}-{run_params['stride']}-{run_params['filter_length']}-{run_params['img_size']}"


def select_unique_corresponding_data_sets(list_of_run_params):
    selected = defaultdict(lambda: False)

    for run_params in list_of_run_params:
        data_key = get_param_data_key(run_params)
        if not selected[data_key]:
            selected[data_key] = run_params

    return list(selected.values())


def parallel_pre_load_datasets(list_of_run_params, processes_per_dataset, device, mnist_path, preencoded_mnist_path):
    list_of_run_params = select_unique_corresponding_data_sets(list_of_run_params)
    #print(f"Multiprocessing of {len(list_of_run_params)} different encoding datasets with {processes} processes.")
    MNISTDataset(0, 0, 0, list_of_run_params[0]['data_shuffle_seed'], path=mnist_path, img_size=list_of_run_params[0]['img_size'])
    threads = []
    for run_params in list_of_run_params:
        thread = QuantumEncodedMNISTOnDiskThread(run_params, device, 10, 10, 10, run_params['data_shuffle_seed'],
                                  path=preencoded_mnist_path, mnist_path=mnist_path, img_size=run_params['img_size'], parallelization=processes_per_dataset)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()



if __name__ == '__main__':
    device = "default.qubit"
    mnist_path = '../mnist'
    preencoded_mnist_path = '../mnist-encoded'
    parallel_pre_load_datasets(circuits, len(os.sched_getaffinity(0)), device, mnist_path, preencoded_mnist_path)
