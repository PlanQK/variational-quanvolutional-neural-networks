import os
import torch


def save_params(model, optim, epoch, save_path):
    """Save model parameters."""
    params = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict()
    }
    path = os.path.join(save_path, "epochs_" + str(epoch) + ".pt")
    torch.save(params, path)


def load_params(model, optimizer, path):
    """Load model parameters."""
    params = torch.load(path)
    model.load_state_dict(params["model_state_dict"])
    optimizer.load_state_dict(params["optim_state_dict"])
    #print("[INFO] loaded the model and optimizer from", path)
    return model, optimizer
