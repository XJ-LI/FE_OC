from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from FunctionEncoder import FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, DistanceCallback
from scipy.io import loadmat

import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--load_path", type=str, default=None)
args = parser.parse_args()

# hyper params
# here we have some predefined values for trained FE
n_basis = 100
train_method = "least_squares"
hidden_size = 256
n_layers = 4
activation = "relu"
arch = "MLP"

# additional hyperparameters
epochs = args.epochs
lr = args.lr
seed = args.seed
load_path = args.load_path
batch_size = args.batch_size

# define a simple MLP for the operator approximation
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=4):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# need to provide the FE model path
if load_path is None:
    raise FileNotFoundError(f"The provided path does not exist: {load_path}")
else:
    logdir = load_path

# seed everything
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

data = loadmat(f'{logdir}/Dataset/quadcopter_dataset_B2B.mat')
target_states = torch.tensor(data['target_states']).to(device)
trajectories = torch.tensor(data['trajectories'][:,:,:50,:].reshape(320, 500, 13)).to(device)
controls = torch.tensor(data['controls'].reshape(320, 500, 4)).to(device)

data = loadmat(f'{logdir}/Dataset/quadcopter_dataset_test.mat')
target_states_val = torch.tensor(data['target_states']).to(device)
trajectories_val = torch.tensor(data['trajectories'][:,:,:50,:].reshape(27, -1, 13)).to(device)
controls_val = torch.tensor(data['controls'].reshape(27, -1, 4)).to(device)

# load the pre-traiend FE model
FE_model = FunctionEncoder(input_size=(13,),
                        output_size=(4,),
                        data_type="deterministic",
                        n_basis=n_basis,
                        model_type=arch,
                        model_kwargs = {"hidden_size": hidden_size, "n_layers": n_layers, "activation": activation},
                        method=train_method).to(device)
FE_model.load_state_dict(torch.load(f"{logdir}/model.pth"))
print('Pretrained Function Encoder Model loaded!')

# define a loos function
def loss_fun(model, FE_model, xs, ys, target_loc):
    representations, _ = FE_model.compute_representation(xs, ys)
    pred_representations = model(target_loc)
    return torch.mean((pred_representations-representations)**2)
    

if __name__ == "__main__":
    # define a MLP for operator learning
    model = MLP(input_dim=3, hidden_dim=256, output_dim=n_basis).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('initial learning rate at: ', lr)

    # Training loop
    for epoch in range(epochs+1):
        # get data
        function_indices = torch.randint(0, 320, (batch_size,), device=device)
        example_xs = trajectories[function_indices]
        example_ys = controls[function_indices]
        target_loc = target_states[function_indices,:3]
        
        xs = trajectories_val
        ys = controls_val
        target_loc_val = target_states_val[:,:3]
        
        optimizer.zero_grad()
        #training loss
        loss_train = loss_fun(model, FE_model, example_xs, example_ys, target_loc)
        # test loss
        with torch.no_grad():
            loss_test = loss_fun(model, FE_model, xs, ys, target_loc_val)
        
        loss_train.backward()
        optimizer.step()
        
        if (epoch+1) % (epochs//50) == 0:
            print(f'Epoch {epoch}, Loss: {loss_train.item()}, Test Loss: {loss_test.item()}')
        
        # reduce learning rate 
        if (epoch+1) % (epochs//4) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
            print('learning rate now set to: ', param_group['lr'])
    
    
    # save the mdoel
    model_save_path = os.path.join(logdir, 'B2B_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"B2B Model saved in directory: {logdir}")  















