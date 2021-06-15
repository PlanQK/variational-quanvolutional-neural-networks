# Variational Quanvolutional Neural Networks with enhanced image encoding
**Preprint**: https://arxiv.org/abs/2106.07327

#### Abstract
Image classification is an important task in various machine learning applications. In recent years, a number of classification methods based on quantum machine learning and different quantum image encoding techniques have been proposed. In this paper, we study the effect of three different quantum image encoding approaches on the performance of a convolution-inspired hybrid quantum-classical image classification algorithm called quanvolutional neural network (QNN). We furthermore examine the effect of variational - i.e. trainable - quantum circuits on the classification results. Our experiments indicate that some image encodings are better suited for variational circuits. However, our experiments show as well that there is not one best image encoding, but that the choice of the encoding depends on the specific constraints of the application. This repository is the official implementation of Variational Quanvolutional Neural Networks with enhanced image encoding. 

## Requirements

For the purpose of reproducibility we use a Docker environment. We provide different options depending on the available resources and the need for development options:

```docker-compose.yml``` for simply running the experiments with the default parameters.

```docker-compose-dev.yml``` same parameters but also makes the code available for quick edits in the docker container.



Additionally, GPUs can be utilized with the following two environments when `"app_params["q-device"] = "qulacs.simulator"`
is set in `main.py`.

```docker-compose-gpu.yml``` for simply running the experiments with the default parameters under utilization of a GPU.

```docker-compose-dev-gpu.yml``` same parameters but also makes the code available within the container to allow for easy execution of modified the code (no rebuild of the image necessary).

<br/>

In the top-level `code` directory run the following commands to set up the container and start bash:

```setup
docker-compose -f docker-compose.yml up -d
docker-compose exec experiments bash
```

From now on all commands are run in this bash shell, unless stated otherwise.

## Training

To train the models in the paper, run the following procedure.

First reproduce our model parameters in yaml files:
```setup_yamls
python generate_experiments.py
```
Then you can run the training procedure with the yaml file corresponding to the desired kernel size and random seed:
```train
python main.py experiments_seed_2_filtersize_2x2_imgsize_14.yaml
```
The code will generate entries in the `save` directory, from which you can inspect the parameters and the performance of the models and reload the pre-trained models. 

The execution of the training can be sped up significantly by pre-encoding the MNIST data and saving the results to disk. To use pre-encoding, change the parameter `app_params["pre_encoding"]` in `main.py` to `True`. **Important:** Pre-encoding works only with the "default.qubit"-simulator. Once the images are pre-encoded, you are free to use any other simulator-backend (e.g. qulacs.simulator).

## Evaluation
You can observe the performance of the models, while they are training or afterwards, through `tensorboard`. Enter this address into your browser, after the experiments have been started, to reach the tensorboard started by default.
```tensorboard
localhost:6006
```
You can also recreate all of the graphs from our paper with the Jupyter Notebook located at:
```evaluation
code/evaluation/Evaluation.ipynb
```
## Testing

You can test models on before unseen data. Simply insert the  save location of the models, you wish to test into the `paths` entry of the `test_params` dictionary in `test_models.py`. For newly trained models this is already performed in `main.py` as specified in the default training parameters.

```test
python test_models.py
```

## Pre-trained Models

All our models can be reproduced using the commands above. However, this is time consuming. We provide pre-trained model checkpoints in the `evaluation/MNIST14x14`-folder.

## Results

These following two tables show the average values of the training, validation and test accuracy grouped by filter size, encoding algorithm and trainability of the quantum circuit and averaged per group over all different random circuits (seeds 0-9).
The mean values for training and validation are additionally averaged over the last 20 epochs in order to better estimate only the effects of trainability and encoding algorithm. The max values are not averaged but belong to a certain model of the specific group over the whole course of training. All models had been tested after 50 epochs of training on the same set of 1000 images.

### Accuracy using 2 x 2 filters
| Encoding              | training       |                | validation     |               | test           |               |
|-----------------------|----------------|----------------|----------------|---------------|----------------|---------------|
|                       | mean           | max            | mean           | max           | mean           | max           |
| FRQI untrainable      | 0.793          | **0.940**          | 0.789          | **0.95**          | 0.806          | 0.878         |
| FRQI trainable        | **0.862**          | 0.935          | **0.853**          | 0.93          | **0.854**          | **0.884**         |
| NEQR untrainable      | 0.785          | 0.875          | 0.773          | 0.93          | 0.760          | 0.830         |
| NEQR trainable        | 0.786          | 0.860          | 0.769          | 0.89          | 0.774          | 0.839         |
| Threshold untrainable | 0.818          | 0.910          | 0.796          | 0.90          | 0.792          | 0.882         |
| Threshold trainable   | 0.827          | 0.925          | 0.796          | 0.92          | 0.825          | 0.881         |

### Accuracy using 4 x 4 filters
| Encoding              | training       |                | validation     |               | test           |               |
|-----------------------|----------------|----------------|----------------|---------------|----------------|---------------|
|                       | mean           | max            | mean           | max           | mean           | max           |
| FRQI untrainable      | 0.639          | 0.870          | 0.623          | 0.84          | 0.602          | 0.826         |
| FRQI trainable        | 0.737          | 0.865          | 0.732          | 0.87          | 0.715          | 0.836         |
| NEQR untrainable      | 0.755          | 0.870          | 0.756          | **0.92**  | 0.727          | 0.789         |
| NEQR trainable        | 0.783          | 0.880          | 0.780          | **0.92** | 0.759          | 0.846         |
| Threshold untrainable | **0.818** | **0.890** | **0.796** | 0.91          | **0.828** | 	**0.859** |
| Threshold trainable   | 0.812          | 0.880          | 0.793          | 0.89          | 0.802          | 0.845         |

<br/>

These now following two tables show the average training duration for each experimental configuration of filter size, encoding and trainability. The training was conducted on a compute cluster with 4 GPUs and 40 CPUs. The quantum states of the input images had been pre-encoded to accelerate the training.

### Training duration in hours using 2 x 2 filters
| Encoding                 | untrainable                                    | trainable |
|--------------------------|------------------------------------------------|-----------|
| FRQI                     | 1.34                                           | 7.78      |
| NEQR                     | 3.51                                           | 13.75     |
| Threshold                | 1.9                                            | 8.19      |
### Training duration in hours using 4 x 4 filters
| Encoding                 | untrainable                                    | trainable |
|--------------------------|------------------------------------------------|-----------|
| FRQI                     | 0.53                                           | 7.31      |
| NEQR                     | 1.55                                           | 12.3      |
| Threshold                | 24.52                                          | 78.32     |

## Contributing
* Denny Mattern - denny.mattern@fokus.fraunhofer.de
* Darya Martyniuk - darya.martyniuk@fokus.fraunhofer.de
* Henri Willems - henri.willems@fokus.fraunhofer.de
* Fabian Bergmann - fabian.bergmann@fokus.fraunhofer.de
* Adrian Paschke - adrian.paschke@fokus.fraunhofer.de


## License and Copyright
Copyright 2021 Denny Mattern, Darya Martyniuk, Henri Willems, Fabian Bergmann and Adrian Paschke.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
