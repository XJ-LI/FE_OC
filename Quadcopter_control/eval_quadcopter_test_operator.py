# new test dataset
## testing a trained model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from FunctionEncoder import FunctionEncoder
from scipy.io import loadmat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

data = loadmat('Dataset/quadcopter_dataset_test.mat')
target_states = data['target_states']; print(target_states.shape)    # 27, 12
trajectories = data['trajectories']; print(trajectories.shape)       # 27, 25, 51, 13
controls = data['controls']; print(controls.shape)                   # 27, 25, 50, 4
T = data['T'].item(); print(T)
N = data['N'].item(); print(N)
g = data['g'].item(); print(g)
m = data['m'].item(); print(m)
dt = T / N

# Indices
POS = slice(0, 3)
VEL = slice(3, 6)
ANG = slice(6, 9)
ANG_VEL = slice(9, 12)
def quad_dynamics(x, u):
    out = torch.zeros_like(x)
    vx, vy, vz = x[VEL]
    psi, theta, phi = x[ANG]
    v_psi, v_theta, v_phi = x[ANG_VEL]
    thrust, tau_psi, tau_theta, tau_phi = u

    out[POS] = torch.stack([vx, vy, vz])
    out[ANG] = torch.stack([v_psi, v_theta, v_phi])

    sin, cos = torch.sin, torch.cos
    ax = (thrust / m) * (sin(psi) * sin(phi) + cos(psi) * sin(theta) * cos(phi))
    ay = (thrust / m) * (-cos(psi) * sin(phi) + sin(psi) * sin(theta) * cos(phi))
    az = (thrust / m) * cos(theta) * cos(phi) - g

    out[VEL] = torch.stack([ax, ay, az])
    out[ANG_VEL] = torch.stack([tau_psi, tau_theta, tau_phi])
    return out


n_basis = 100
hidden_size = 256
n_layers = 4
activation = "relu"
train_method = "least_squares"

model = FunctionEncoder(input_size=(13,),
                        output_size=(4,),
                        data_type="deterministic",
                        n_basis=n_basis,
                        model_type="MLP",
                        model_kwargs = {"hidden_size": hidden_size, "n_layers": n_layers, "activation": activation},
                        method=train_method).to(device)
model.load_state_dict(torch.load(f"model.pth"))

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

B2B_model = MLP(input_dim=3, hidden_dim=256, output_dim=n_basis).to(device)
B2B_model.load_state_dict(torch.load(f"B2B_model.pth"))

# error calculation test dataset
obj_pred = 0
obj_true = 0
for idx in range(target_states.shape[0]):
    print('test target: ', target_states[idx][:3])
    target_idx = idx
    target_loc = target_loc = torch.tensor(target_states[idx, :3].reshape(1, 3)).to(device)
    representation = B2B_model(target_loc)

    x = torch.tensor(trajectories[[target_idx], 5:, 0, :12])
    x_pred_all = torch.zeros(20, 51, 12)
    u_pred_all = torch.zeros(20, 50, 4)
    x_pred_all[:,0,:] = x

    for i in range(N):
        xt = torch.cat([x, dt*i*torch.ones(1, 20, 1)], dim=-1).to(device)
        u_pred = model.predict(xt, representation).cpu().detach()
        u_pred_all[:,i,:] = u_pred
        for j in range(20):
            update_j = quad_dynamics(x[:,j,:].flatten(), u_pred[:,j,:].flatten())
            x[:,j,:] = x[:,j,:] + dt * update_j
        x_pred_all[:,i+1,:] = x
    # calculate cost
    terminal_cost = 0.5 * 1000.0 * torch.norm(x_pred_all[:,-1,:] - target_states[[idx],:], dim=1)**2
    control_cost = 0.5 * dt * (u_pred_all ** 2).sum(dim=2).sum(dim=1) 
    avg_cost = torch.mean(terminal_cost + control_cost)
    obj_pred += avg_cost


    # now do it with the ground truth
    x_true_all = torch.tensor(trajectories[idx, 5:, :, :12]).reshape(20, 51, 12)
    u_true_all = torch.tensor(controls[idx, 5:, :, :]).reshape(20, 50, 4)
    terminal_cost = 0.5 * 1000.0 * torch.norm(x_true_all[:,-1,:] - target_states[[idx],:], dim=1)**2
    control_cost = 0.5 * dt * (u_true_all ** 2).sum(dim=2).sum(dim=1) 
    avg_cost = torch.mean(terminal_cost + control_cost)
    obj_true += avg_cost

# final averging, over different target states
obj_pred = obj_pred / target_states.shape[0]
obj_true = obj_true / target_states.shape[0]
print('Ground Truth -- average objective function value over multiple targets and multiple initial states: ', obj_true.item())
print('Prediction -- average objective function value over multiple targets and multiple initial states: ', obj_pred.item())

