from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat

from FunctionEncoder import FunctionEncoder
from OCDataset import OCDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

# load a existing model
n_basis = 100
train_method = "least_squares"
seed = 0
load_path = "./"
arch = "MLP"


# helper function for plotting
def Q(x):
    x = np.atleast_2d(x)
    cov_inv = np.eye(2) / 0.4
    quad = np.sum(x @ cov_inv * x, axis=1)
    return 50 * np.exp(-0.5 * quad)


dataset = OCDataset(device=device)
model = FunctionEncoder(input_size=dataset.input_size,
                        output_size=dataset.output_size,
                        data_type=dataset.data_type,
                        n_basis=n_basis,
                        model_type="MLP",
                        method=train_method).to(device)
model.load_state_dict(torch.load(f"{load_path}/model.pth", map_location=device))


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


B2B_model = MLP(input_dim=2, hidden_dim=256, output_dim=n_basis).to(device)
B2B_model.load_state_dict(torch.load(f"{load_path}/B2B_model.pth", map_location=device))


tests = [
    # name, mat_path, idx_for_plot, n_cases
    ("seen",    "Dataset/oc_velocity_dataset_seen.mat",    11, 16),
    ("unseen1", "Dataset/oc_velocity_dataset_unseen1.mat", 0,  5),
    ("unseen2", "Dataset/oc_velocity_dataset_unseen2.mat", 0,  5),
]

left = -3
right = 2.5
x1 = np.linspace(left, right, 200)
x2 = np.linspace(left, right, 200)
X1, X2 = np.meshgrid(x1, x2)
grid = np.column_stack([X1.ravel(), X2.ravel()])
Z = Q(grid).reshape(X1.shape)

for name, mat_path, idx, n_cases in tests:
    # seed torch 
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 80)
    print(f"Running test: {name} ({mat_path})")

    data = loadmat(mat_path)
    xs = torch.tensor(data['positions_times'], device=device, dtype=torch.float)           # (n_cases, 40, 20, 3)
    ys = torch.tensor(data['velocities'], device=device, dtype=torch.float)                # (n_cases, 40, 20, 2)
    target_states = torch.tensor(data['target_states'], device=device, dtype=torch.float)  # (n_cases, 2)


    target_loc = target_states[idx, :].reshape(1, 2)
    representation = B2B_model(target_loc)

    n_traj = 20
    dt = 1/20

    plt.figure(figsize=(6, 6))
    plt.imshow(Z, extent=[left, right, left, right], origin='lower', cmap='viridis')

    xs_test = xs[idx, :n_traj, 0, :].reshape(1, n_traj, 3)
    plt.scatter(xs_test[0, :, 0].detach().cpu().numpy(),
                xs_test[0, :, 1].detach().cpu().numpy(),
                marker='o', color='grey', label='True')
    plt.scatter(xs_test[0, :, 0].detach().cpu().numpy(),
                xs_test[0, :, 1].detach().cpu().numpy(),
                marker='x', color='red', label='Pred')

    xs_true = xs_test[:, :, :2]
    xs_pred = xs_test[:, :, :2]
    for t in range(20):
        ys_true = ys[idx, :n_traj, t, :].reshape(1, n_traj, 2)
        ys_pred = model.predict(xs_test, representation)

        xs_true = xs_true + dt * ys_true
        xs_pred = xs_pred + dt * ys_pred

        xs_test = torch.cat((xs_pred, torch.full_like(xs_pred[:, :, :1], dt*(t+1))), dim=-1)
        plt.scatter(xs_true[0, :, 0].detach().cpu().numpy(),
                    xs_true[0, :, 1].detach().cpu().numpy(),
                    marker='o', color='grey', label=None)
        plt.scatter(xs_pred[0, :, 0].detach().cpu().numpy(),
                    xs_pred[0, :, 1].detach().cpu().numpy(),
                    marker='x', color='red', label=None)

    if name == "seen":
        for i in range(16):
            if i != idx:
                plt.scatter(target_states[i, 0].detach().cpu().numpy(),
                            target_states[i, 1].detach().cpu().numpy(),
                            color='grey', s=200, marker='*', label=None)
            else:
                plt.scatter(target_states[idx, 0].detach().cpu().numpy(),
                            target_states[idx, 1].detach().cpu().numpy(),
                            color='blue', s=200, marker='*', label='Target')
        title_y = target_states[idx, 1].detach().cpu().numpy()
        plt.title(f'State Trajectories (target state: [{target_states[idx,0].detach().cpu().numpy():.2f}, {title_y:.2f}])', fontsize=16)
    else:
        x_coords, y_coords = np.meshgrid(np.linspace(1, 2, 4), np.linspace(1, 2, 4))
        grid_points = np.column_stack((x_coords.flatten(), y_coords.flatten()))
        for i in range(16):
            plt.scatter(grid_points[i, 0], grid_points[i, 1], color='grey', s=200, marker='*', label=None)
        plt.scatter(target_states[idx, 0].detach().cpu().numpy(),
                    target_states[idx, 1].detach().cpu().numpy(),
                    color='blue', s=200, marker='*', label='Target')
        plt.title(f'State Trajectories (target state: [{target_states[idx,0].detach().cpu().numpy():.2f}, {target_states[idx,1].detach().cpu().numpy():.2f}])', fontsize=16)

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xlim([left, right])
    plt.ylim([left, right])
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"oc_test_b2b_{name}.png", bbox_inches='tight', dpi=400)
    plt.close()

    # objective evaluation
    total_obj_true = 0
    total_obj_pred = 0
    for i in range(n_cases):
        case_obj_true = 0
        case_obj_pred = 0

        representation = B2B_model(target_states[i, :].reshape(1, 2))

        xs_true_full = xs[i, :, 0, :].reshape(1, 40, 3)
        xs_pred_full = xs[i, :, 0, :].reshape(1, 40, 3)
        xs_true = xs_true_full[:, :, :2]
        xs_pred = xs_pred_full[:, :, :2]

        for j in range(20):
            case_obj_true += dt * Q(xs_true.reshape(40, 2).detach().cpu().numpy())
            case_obj_pred += dt * Q(xs_pred.reshape(40, 2).detach().cpu().numpy())

            ys_true = ys[i, :, j, :].reshape(1, 40, 2)
            ys_pred = model.predict(xs_pred_full, representation)

            case_obj_true += 0.5 * dt * torch.norm(ys_true.squeeze(0), dim=1).detach().cpu().numpy()
            case_obj_pred += 0.5 * dt * torch.norm(ys_pred.squeeze(0), dim=1).detach().cpu().numpy()

            xs_true = xs_true + dt * ys_true
            xs_pred = xs_pred + dt * ys_pred
            xs_true_full = torch.cat((xs_true, torch.full_like(xs_true[:, :, :1], dt*(j+1))), dim=-1)
            xs_pred_full = torch.cat((xs_pred, torch.full_like(xs_pred[:, :, :1], dt*(j+1))), dim=-1)

        # terminal state
        case_obj_true += 50 * np.sum(((xs_true - target_states[i]).reshape(40, 2).detach().cpu().numpy())**2, axis=1)
        case_obj_pred += 50 * np.sum(((xs_pred - target_states[i]).reshape(40, 2).detach().cpu().numpy())**2, axis=1)

        print(f"terminal state: {target_states[i].detach().cpu().numpy()}, average true objective loss: {case_obj_true.mean().item()}, average pred objective loss: {case_obj_pred.mean().item()}")
        total_obj_true += case_obj_true.mean().item()
        total_obj_pred += case_obj_pred.mean().item()

    print(f"[{name}] combined average true objective loss across all sub-problems: {total_obj_true/n_cases}, combined average pred objective loss across all sub-problems: {total_obj_pred/n_cases}")
