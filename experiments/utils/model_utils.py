import os

import pandas as pd


def find_final(paths):
    max_model_num = -1
    max_model = ""

    for path in paths:

        model_file = os.path.split(path)[-1]
        model_num = int(''.join(filter(str.isdigit, model_file)))

        if model_num > max_model_num:
            max_model_num = model_num
            max_model = path

    return max_model


def find_best_val(paths):

    train_results_path = get_closest_file(paths[0], file_names=["train_result.csv"])
    train_results = pd.read_csv(train_results_path)
    pos_max = train_results['val_loss'].argmin()
    epoch_max = train_results['epoch'][pos_max]

    for path in paths:
        model_file = os.path.split(path)[-1]
        model_num = int(''.join(filter(str.isdigit, model_file)))
        if model_num == epoch_max:
            return path

    raise ValueError(f"No model path corresponding to epoch {epoch_max} found")


def get_models_in_dir(dir_path, best_val_loss=True):
    files = []
    same_dir_models = []
    for content in os.listdir(dir_path):

        content_path = os.path.join(dir_path, content)

        if os.path.isdir(content_path):
            files += get_models_in_dir(content_path)

        elif os.path.isfile(content_path):
            if content.split(".")[-1] == "pt":
                same_dir_models.append(content_path)

    if same_dir_models:
        if best_val_loss:
            model_path = find_best_val(same_dir_models)
        else:
            model_path = find_final(same_dir_models)
        if model_path != "":
            files.append(model_path)
    return files


def get_closest_file(file_path, file_names):
    path = os.path.split(file_path)[0]
    while path != "":
        for content in os.listdir(path):
            for file_name in file_names:
                if content == file_name:
                    return os.path.join(path, file_name)
        path = os.path.split(path)[0]

    raise ValueError(f"No files called like {file_names} found in path {file_path}")


def get_model_and_yaml_paths(paths, best_val_loss_model=True):
    models = []
    for path in paths:
        if os.path.isdir(path):
            models += get_models_in_dir(path, best_val_loss=best_val_loss_model)
        elif os.path.isfile(path) and path.split('.')[-1] == 'pt':
            models.append(path)
        else:
            raise ValueError(f"No model or directory found at path {path}")

    return [(model_path, get_closest_file(model_path, file_names=["run-params.yaml", "hyper-params.yaml"])) for model_path in models]
