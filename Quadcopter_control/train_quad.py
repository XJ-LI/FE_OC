from datetime import datetime

import matplotlib.pyplot as plt
import torch
import numpy as np

from FunctionEncoder import FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, DistanceCallback
from QuadDataset import QuadDataset

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=100000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--arch", type=str, default="MLP")
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
lr = args.lr
train_method = args.train_method
seed = args.seed
load_path = args.load_path
arch = args.arch
hidden_size = args.hidden_size
n_layers = args.n_layers
activation = args.activation

if load_path is None:
    logdir = f"logs_Quad_{epochs}_{hidden_size}_{n_layers}_{activation}_{lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)
dataset = QuadDataset()

if load_path is None:
    # create the model
    if arch == "MLP":
        model = FunctionEncoder(input_size=dataset.input_size,
                                output_size=dataset.output_size,
                                data_type=dataset.data_type,
                                n_basis=n_basis,
                                model_type=arch,
                                model_kwargs = {"hidden_size": hidden_size, "n_layers": n_layers, "activation": activation},
                                method=train_method).to(device)
    else: 
        raise ValueError(f"Unsupported arch_type: {arch}.")  # assume MLP only for now
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    if lr != 0.001:
        for param_group in model.opt.param_groups:
            param_group['lr'] = lr
        print('learning rate set to: ', lr)

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
    print(f'trained model saved to {logdir}')
else:
    # load the model
    if arch == "MLP":
        model = FunctionEncoder(input_size=dataset.input_size,
                                output_size=dataset.output_size,
                                data_type=dataset.data_type,
                                n_basis=n_basis,
                                model_type="MLP",
                                method=train_method).to(device)
        model.load_state_dict(torch.load(f"{logdir}/model.pth"))
    else:
        raise ValueError(f"Unsupported arch_type: {arch}.")
        