## testing a trained model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from FunctionEncoder import FunctionEncoder
from scipy.io import loadmat
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

data = loadmat('Dataset/bicycle_dataset_2_shuffled.mat')

# NOTE: keep your "last 32" slicing
trajectories = data['state_time'][-32:, :, :, :]   # (32, 100, 51, 5) expected
print(trajectories.shape)
controls    = data['controls'][-32:, :, :, :]      # (32, 100, 50, 2) expected
print(controls.shape)

T = data['T'].item();       print(T)
N = data['N'].item();       print(N)   # expect 50
L = data['L'].item();       print(L)
alpha_G = data['alpha_G'].item(); print(alpha_G)
dt = T / N

# load pre-trained function encoder model
n_basis = 200
hidden_size = 256
n_layers = 6
activation = "relu"
train_method = "least_squares"

model = FunctionEncoder(input_size=(5,),
                        output_size=(2,),
                        data_type="deterministic",
                        n_basis=n_basis,
                        model_type="MLP",
                        model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers, "activation": activation},
                        method=train_method).to(device)
model.load_state_dict(torch.load(f"model_2.pth", map_location=device))
model.eval()

# define a forward model (numpy step)
def bicycle_dynamics(z, u):
    """Continuous-time kinematic bicycle dynamics.
    z = [x, y, theta, v], u = [delta, a]
    """
    x, y, th, v = z
    delta, a = u
    xdot  = v * np.cos(th)
    ydot  = v * np.sin(th)
    thdot = (v / L) * np.tan(delta)
    vdot  = a
    return np.array([xdot, ydot, thdot, vdot], dtype=float)

# ---------------- EVALUATION FIRST ----------------
num_inference = 50  # number of trajectories used for coefficient calculation
n_cases = trajectories.shape[0]

# make z_target once
z_target = torch.tensor(data['z_target'].flatten(), dtype=torch.float32)  # shape (4,)

pred_costs = []
true_costs = []
control_costs_pred = []
control_costs_true = []
obstacle_costs_pred = []
obstacle_costs_true = []
terminal_deviation_pred = []
terminal_deviation_true = []
success_indices = []
failed_indices = []


with torch.no_grad():
    for idx in range(n_cases):
        # indexing trick: parameters live at +544 relative to this sliced block
        idx_param = idx + 544
        A1 = float(np.array(data['A1_cases'].flatten()[idx_param]))
        mu1_np = np.array(data['mu1_cases'][idx_param]).reshape(-1)  # (2,)
        sigma1 = float(np.array(data['sigma1_cases'].flatten()[idx_param]))
        A2 = float(np.array(data['A2_cases'].flatten()[idx_param]))
        mu2_np = np.array(data['mu2_cases'][idx_param]).reshape(-1)  # (2,)
        sigma2 = float(np.array(data['sigma2_cases'].flatten()[idx_param]))

        mu1_t = torch.tensor(mu1_np, dtype=torch.float32)
        mu2_t = torch.tensor(mu2_np, dtype=torch.float32)

        # representation from first num_inference trajectories
        example_xs = torch.tensor(trajectories[idx, :num_inference, :50, :]).reshape(1, num_inference*50, 5).float().to(device)
        example_ys = torch.tensor(controls[idx, :num_inference, :, :]).reshape(1, num_inference*50, 2).float().to(device)
        representation, _ = model.compute_representation(example_xs, example_ys, method="least_squares")

        # rollout predicted for the remaining trajectories
        n_roll = 100 - num_inference
        x = torch.tensor(trajectories[[idx], num_inference:, 0, :4])           # (1, n_roll, 4) CPU
        x_pred_all = torch.zeros(n_roll, 51, 4)
        u_pred_all = torch.zeros(n_roll, 50, 2)
        x_pred_all[:, 0, :] = x
        x_true_all = torch.tensor(trajectories[idx, num_inference:, :, :4]).reshape(n_roll, 51, 4)
        u_true_all = torch.tensor(controls[idx, num_inference:, :, :]).reshape(n_roll, 50, 2)

        obstacle_cost_pred = torch.zeros(n_roll)
        obstacle_cost_true = torch.zeros(n_roll)

        for i in range(N):
            xt = torch.cat([x, dt*i*torch.ones(1, n_roll, 1)], dim=-1).float().to(device)
            u_pred = model.predict(xt, representation).detach().cpu()          # (1, n_roll, 2)
            u_pred_all[:, i, :] = u_pred[0]

            # integrate dynamics per trajectory (CPU + numpy step)
            for j in range(n_roll):
                update_j = bicycle_dynamics(x[:, j, :].flatten().numpy(), u_pred[:, j, :].flatten().numpy())
                x[:, j, :] = x[:, j, :] + dt * torch.tensor(update_j, dtype=torch.float32)

            x_pred_all[:, i+1, :] = x

            # obstacle costs (two Gaussians)
            sq1 = ((x[0, :, :2] - mu1_t) ** 2).sum(dim=1)
            sq2 = ((x[0, :, :2] - mu2_t) ** 2).sum(dim=1)
            obstacle_cost_pred += 0.5 * dt * (A1 * torch.exp(-0.5 * sq1 / (sigma1 ** 2))
                                              + A2 * torch.exp(-0.5 * sq2 / (sigma2 ** 2)))

            sq1_t = ((x_true_all[:, i, :2] - mu1_t) ** 2).sum(dim=1)
            sq2_t = ((x_true_all[:, i, :2] - mu2_t) ** 2).sum(dim=1)
            obstacle_cost_true += 0.5 * dt * (A1 * torch.exp(-0.5 * sq1_t / (sigma1 ** 2))
                                              + A2 * torch.exp(-0.5 * sq2_t / (sigma2 ** 2)))
            
        terminal_cost_pred = 0.5 * alpha_G * torch.norm(x_pred_all[:, -1, :] - z_target.reshape(1, 4), dim=1) ** 2
        control_cost_pred  = 0.5 * dt * (u_pred_all ** 2).sum(dim=(1, 2))
        total_cost_pred    = terminal_cost_pred + control_cost_pred + obstacle_cost_pred  # (50,)

        terminal_cost_true = 0.5 * alpha_G * torch.norm(x_true_all[:, -1, :] - z_target.reshape(1, 4), dim=1) ** 2
        control_cost_true  = 0.5 * dt * (u_true_all ** 2).sum(dim=(1, 2))
        total_cost_true    = terminal_cost_true + control_cost_true + obstacle_cost_true  # (50,)


        avg_pred = torch.mean(total_cost_pred).item()
        avg_true = torch.mean(total_cost_true).item()

        pred_costs.append(avg_pred)
        true_costs.append(avg_true)



        control_costs_pred.append(torch.mean(control_cost_pred).item())
        control_costs_true.append(torch.mean(control_cost_true).item())
        obstacle_costs_pred.append(torch.mean(obstacle_cost_pred).item())
        obstacle_costs_true.append(torch.mean(obstacle_cost_true).item())

        term_dev_pred = torch.norm(x_pred_all[:, -1, :2] - z_target[:2].reshape(1, 2), dim=1) ** 2
        term_dev_true = torch.norm(x_true_all[:, -1, :2] - z_target[:2].reshape(1, 2), dim=1) ** 2
        terminal_deviation_pred.append(torch.mean(term_dev_pred).item())
        terminal_deviation_true.append(torch.mean(term_dev_true).item())

        # classify
        if avg_pred <= 2.0 * avg_true:
            success_indices.append(idx)
        else:
            failed_indices.append(idx)

        print(f"[case {idx:02d}] pred_mean={avg_pred:.6f}  true_mean={avg_true:.6f} -> {'SUCCESS' if idx in success_indices else 'FAIL'}")

pred_costs = np.array(pred_costs, dtype=float)
true_costs = np.array(true_costs, dtype=float)

# report summary
n_success = len(success_indices)
n_failed = len(failed_indices)
print("\n=== Evaluation Summary ===")
print(f"Total cases: {n_cases}")
print(f"Success: {n_success}")
print(f"Fail:    {n_failed}")

if n_success > 0:
    s_mask = np.zeros(n_cases, dtype=bool)
    s_mask[success_indices] = True
    print(f"Avg (successful only) — pred: {pred_costs[s_mask].mean():.6f}, true: {true_costs[s_mask].mean():.6f}")
else:
    print("No successful cases — skipping success averages.")

print(f"average predictive control cost across all subproblems: {np.mean(control_costs_pred):.6f}")
print(f"average True control cost across all subproblems: {np.mean(control_costs_true):.6f}")
print(f"average predictive state cost across all subproblems: {np.mean(obstacle_costs_pred):.6f}")
print(f"average True state cost across all subproblems: {np.mean(obstacle_costs_true):.6f}")
print(f"average predictive terminal state deviation across all subproblems: {np.mean(terminal_deviation_pred):.6f}")
print(f"average true terminal state deviation across all subproblems: {np.mean(terminal_deviation_true):.6f}")

# ---------------- PLOTTING ----------------
outdir = "./plots_eval_2_obs"
os.makedirs(outdir, exist_ok=True)

def rerollout_case(idx):
    """Recompute prediction rollout and return (x_pred, x_true, u_pred_all, u_true_all, A1, mu1_np, sigma1, A2, mu2_np, sigma2) on CPU.
       NOTE: Uses DOUBLE-GAUSSIAN params with idx+544 (matching your evaluation code convention).
    """
    # ---- CHANGED: fetch double-obstacle params with idx + 544 ----
    idx_param = idx + 544
    A1     = float(np.array(data['A1_cases'].flatten()[idx_param]))
    mu1_np = np.array(data['mu1_cases'][idx_param]).reshape(-1)  # (2,)
    sigma1 = float(np.array(data['sigma1_cases'].flatten()[idx_param]))
    A2     = float(np.array(data['A2_cases'].flatten()[idx_param]))
    mu2_np = np.array(data['mu2_cases'][idx_param]).reshape(-1)  # (2,)
    sigma2 = float(np.array(data['sigma2_cases'].flatten()[idx_param]))

    # representation from first num_inference
    example_xs = torch.tensor(trajectories[idx, :num_inference, :50, :]).reshape(1, num_inference*50, 5).float().to(device)
    example_ys = torch.tensor(controls[idx, :num_inference, :, :]).reshape(1, num_inference*50, 2).float().to(device)
    with torch.no_grad():
        representation, _ = model.compute_representation(example_xs, example_ys, method="least_squares")

    # rollout for remaining
    x = torch.tensor(trajectories[[idx], num_inference:, 0, :4])    # (1, M, 4)
    M = x.shape[1]  # typically 50
    x_pred = torch.zeros(M, 51, 4)
    u_pred_all = torch.zeros(M, 50, 2)
    x_pred[:, 0, :] = x
    x_true = torch.tensor(trajectories[idx, num_inference:, :, :4]) # (M, 51, 4)
    u_true_all = torch.tensor(controls[idx, num_inference:, :, :])  # (M, 50, 2)

    with torch.no_grad():
        for i in range(N):
            xt = torch.cat([x, dt*i*torch.ones(1, M, 1)], dim=-1).float().to(device)
            u_pred = model.predict(xt, representation).detach().cpu()  # (1, M, 2)
            u_pred_all[:, i, :] = u_pred[0]
            # step dynamics (CPU)
            for j in range(M):
                update_j = bicycle_dynamics(x[:, j, :].flatten().numpy(), u_pred[:, j, :].flatten().numpy())
                x[:, j, :] = x[:, j, :] + dt * torch.tensor(update_j, dtype=torch.float32)
            x_pred[:, i+1, :] = x

    return x_pred, x_true, u_pred_all, u_true_all, A1, mu1_np, sigma1, A2, mu2_np, sigma2

def plot_case_3x1(idx, tag):
    """3×1 figure: (1) Pred trajs, (2) True trajs, (3) Controls (5 scenarios)."""
    # ---- CHANGED: unpack double-obstacle params ----
    (x_pred, x_true, u_pred_all, u_true_all,
     A1, mu1_np, sigma1, A2, mu2_np, sigma2) = rerollout_case(idx)
    mu1_x, mu1_y = float(mu1_np[0]), float(mu1_np[1])
    mu2_x, mu2_y = float(mu2_np[0]), float(mu2_np[1])

    # grid for obstacle contour (sum of two Gaussians)
    gx = np.linspace(-1.5, 6.5, 240)
    gy = np.linspace(-1.5, 6.5, 240)
    GX, GY = np.meshgrid(gx, gy)
    GQ = (
        A1 * np.exp(-0.5 * ((GX - mu1_x)**2 + (GY - mu1_y)**2) / (sigma1**2))
      + A2 * np.exp(-0.5 * ((GX - mu2_x)**2 + (GY - mu2_y)**2) / (sigma2**2))
    )

    # ---- EXACT same figure sizing & labeling style you used ----
    fig, axes = plt.subplots(3, 1, figsize=(4, 12))
    ax_pred, ax_true, ax_ctrl = axes

    # ----- (1) Prediction trajectories -----
    cs1 = ax_pred.contour(GX, GY, GQ, levels=12)
    ax_pred.clabel(cs1, inline=True, fontsize=8)
    M = x_pred.shape[0]
    for i in range(M):
        ax_pred.plot(x_pred[i, :, 0].numpy(), x_pred[i, :, 1].numpy(), lw=2)
    ax_pred.plot(z_target[0].item(), z_target[1].item(), marker='*', ms=12)
    ax_pred.text(z_target[0].item()+0.1, z_target[1].item()+0.1, "target")
    ax_pred.set_title(f"Case {idx:02d} — Prediction", fontsize = 18)
    ax_pred.set_xlabel("x", fontsize = 16)
    ax_pred.set_ylabel("y", fontsize = 16)
    ax_pred.set_aspect("equal", adjustable="box")
    ax_pred.grid(False)

    # ----- (2) Ground-truth trajectories -----
    cs2 = ax_true.contour(GX, GY, GQ, levels=12)
    ax_true.clabel(cs2, inline=True, fontsize=8)
    for i in range(M):
        ax_true.plot(x_true[i, :, 0].numpy(), x_true[i, :, 1].numpy(), '--', lw=2)
    ax_true.plot(z_target[0].item(), z_target[1].item(), marker='*', ms=12)
    ax_true.text(z_target[0].item()+0.1, z_target[1].item()+0.1, "target")
    ax_true.set_title(f"Case {idx:02d} — Ground Truth", fontsize = 18)
    ax_true.set_xlabel("x", fontsize = 16)
    ax_true.set_ylabel("y", fontsize = 16)
    ax_true.set_aspect("equal", adjustable="box")
    ax_true.grid(False)

    # ----- (3) Controls plot (ONLY 5 scenarios) -----
    M = u_pred_all.shape[0]
    n_show = min(5, M)  # CHANGED: 5 scenarios as requested
    show_ids = np.array(range(n_show))  # keep your simple indexing
    t_steps = np.linspace(0, T, N)      # keep your time axis formatting
    for j in show_ids:
        # control dim 0
        ax_ctrl.plot(t_steps, u_pred_all[j, :, 0].numpy(), 'r-', alpha=0.9, linewidth=1.8)
        ax_ctrl.plot(t_steps, u_true_all[j, :, 0].numpy(), 'r:', alpha=0.9, linewidth=1.8)
        # control dim 1
        ax_ctrl.plot(t_steps, u_pred_all[j, :, 1].numpy(), 'b-', alpha=0.9, linewidth=1.8)
        ax_ctrl.plot(t_steps, u_true_all[j, :, 1].numpy(), 'b:', alpha=0.9, linewidth=1.8)

    # legend (proxy handles to avoid duplicates) — unchanged
    proxy_lines = [
        plt.Line2D([0], [0], linestyle='-', color='r', linewidth=2, label='$\\delta$ pred'),
        plt.Line2D([0], [0], linestyle=':', color='r', linewidth=2, label='$\\delta$ true'),
        plt.Line2D([0], [0], linestyle='-', color='b', linewidth=2, label='a pred'),
        plt.Line2D([0], [0], linestyle=':', color='b', linewidth=2, label='a true'),
    ]
    ax_ctrl.legend(handles=proxy_lines, loc='best', fontsize=9)
    ax_ctrl.set_title(f"Case {idx:02d} — Controls", fontsize = 18)
    ax_ctrl.set_box_aspect(1)
    ax_ctrl.set_xlabel("time", fontsize = 16)
    ax_ctrl.set_ylabel("value", fontsize = 16)
    ax_ctrl.grid(True)

    plt.tight_layout()
    fname = os.path.join(outdir, f"case_{idx:02d}_{tag}_3x1.pdf")
    fig.savefig(fname, bbox_inches="tight")
    # plt.show()
    plt.close(fig)
    print(f"Saved: {fname}")

# ---- select top-3 by predicted average cost (descending) and plot ----
top3_idx = np.argsort(-pred_costs)[:3]
print("Top-3 cases by predicted average cost:", top3_idx.tolist())

for idx in top3_idx:
    plot_case_3x1(idx, tag="top3_predcost")
