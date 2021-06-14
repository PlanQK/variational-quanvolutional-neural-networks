import yaml

hyper_params = {
    "out_channels": 4,  # depends on the encoding
    "circuit_layers": 1,
    "n_rotations": 4,
    "filter_length": 2,
    "stride": 1,
    "out_features": 10,
    "batch_size": 2,
    "epochs_num": 50,
    "steps_in_epoch": 100,
    "val_data_size": 50,
    "train_split_percent": 0.8,
    "run_test": True,
    "data": 'MNIST',
    "img_size": 14, # use 28x28 images; change for resize

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
    "trainable": True,
    "train_samples": 10000,
    "valid_samples": 200,
    "test_samples": 1000,

}

TRESHOLD_2x2_params = {
    "encoder": "Threshold_Encoder",
    "filter_length": 2,
    "out_channels": 4,  # depends on the encoding
}

TRESHOLD_4x4_params = {
    "encoder": "Threshold_Encoder",
    "filter_length": 4,
    "out_channels": 16,  # depends on the encoding
    "n_rotations": 10,
    "stride": 2
}

NEQR_2x2_params = {
    "encoder": "NEQR",
    "filter_length": 2,
    "out_channels": 10,  # depends on the encoding
}

NEQR_4x4_params = {
    "encoder": "NEQR",
    "filter_length": 4,
    "out_channels": 12,  # depends on the encoding
    "n_rotations": 10,
    "stride": 2
}

FRQI_2x2_params = {
    "encoder": "FRQI_for_2x2",
    "filter_length": 2,
    "out_channels": 3,  # depends on the encoding
}

FRQI_4x4_params = {
    "encoder": "FRQI_for_4x4",
    "filter_length": 4,
    "out_channels": 8,  # depends on the encoding
    "n_rotations": 10,
    "stride": 2
}


trainables = [True, False]

encoders = {
    "2x2": [
        TRESHOLD_2x2_params,
        NEQR_2x2_params,
        FRQI_2x2_params,
    ],

    "4x4": [
        TRESHOLD_4x4_params,
        NEQR_4x4_params,
        FRQI_4x4_params,
    ]
}

seeds = list(range(10))



for seed in seeds:
    for key in encoders.keys():

        experiments = {}

        for encoder in encoders[key]:
            for trainable in trainables:

                params = hyper_params.copy()
                params.update(encoder)
                params["trainable"] = trainable
                params["calculation_seed"] = seed

                params["calculation_args"] = {}
                params["encoder_args"] = {}
                params["measurement_args"] = {}

                name = "Seed_{}_Trainable_{}_{}_{}_imgsize_{}".format(seed, "yes" if trainable else "no", encoder["encoder"], key, params["img_size"])

                experiments[name] = params

                print(name)

        with open("experiments_seed_{}_filtersize_{}_imgsize_{}.yaml".format(seed, key, params["img_size"]), "w") as f:
            result = yaml.dump(experiments, f)
