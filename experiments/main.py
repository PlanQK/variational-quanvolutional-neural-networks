import logging
import os, sys, time
import warnings

warnings.filterwarnings('ignore')
import pennylane as qml
import shutil
import pennylane_qulacs

import yaml
from datetime import datetime

module_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(module_dir)

from runners import ExperimentRunner
from utils.parallel import run_experiments_parallel

app_params = {
    "logs_path": './save/experiments',
    "logging_level": logging.DEBUG,
    "multiprocessing": False,
    "processes": len(os.sched_getaffinity(0)),
    "save_preloaded": False,
    "pre_encoding": False,
    "q-device": "default.qubit"
    # "q-device": "qulacs.simulator"
    # "q-device": "default.qubit.autograd"
    # "q-device": "lightning.qubit"
    # "q-device": "qiskit.ibmq"
}


if __name__ == '__main__':

    root_dir = str(app_params["logs_path"] + '_date_' +
                   datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

    # if experiments are defined in yaml-file, use these
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as experiments_file:
            experiments_definitions = yaml.load(experiments_file)

        experiment_names = list(experiments_definitions.keys())
        circuits = [experiments_definitions[key] for key in experiments_definitions.keys()]

    runners = []
    for circuit_params, exp_name in zip(circuits, experiment_names):
        # print(circuit_params, exp_name)
        circuit_params['name'] = exp_name
        exp_dir = os.path.join(root_dir, exp_name)
        runners.append(
            ExperimentRunner(circuit_params, exp_dir, app_params["q-device"], app_params['logging_level'], app_params['save_preloaded'], app_params['pre_encoding']))

    start = time.time()
    if app_params['multiprocessing']:
        results = run_experiments_parallel(runners, app_params['processes'])
    else:
        results = []
        for runner in runners:
            results.append(runner.run())

    print(f"Finished experiments after {(time.time() - start) * 1000}ms")
    print(results)

