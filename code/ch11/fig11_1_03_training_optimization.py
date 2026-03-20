"""fig11_1_03_training_optimization.py
优化器对比：SGD、Momentum、Adam 在 Beale 函数上的收敛轨迹"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

# ── Beale function & gradient ────────────────────────────────────────────────

def beale(x, y):
    """Beale function — minimum at (3, 0.5)."""
    return ((1.5 - x + x * y) ** 2
            + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def grad_beale(x, y, eps=1e-7):
    """Central-difference numerical gradient of the Beale function."""
    dx = (beale(x + eps, y) - beale(x - eps, y)) / (2 * eps)
    dy = (beale(x, y + eps) - beale(x, y - eps)) / (2 * eps)
    return np.array([dx, dy])


# ── Optimizers ───────────────────────────────────────────────────────────────

def run_sgd(start, lr, n_steps):
    """Vanilla SGD: theta -= lr * grad."""
    theta = np.array(start, dtype=float)
    trajectory = [theta.copy()]
    for _ in range(n_steps):
        g = grad_beale(theta[0], theta[1])
        theta = theta - lr * g
        trajectory.append(theta.copy())
    return np.array(trajectory)


def run_momentum(start, lr, gamma, n_steps):
    """SGD with momentum: v = gamma*v + lr*grad; theta -= v."""
    theta = np.array(start, dtype=float)
    v = np.zeros(2)
    trajectory = [theta.copy()]
    for _ in range(n_steps):
        g = grad_beale(theta[0], theta[1])
        v = gamma * v + lr * g
        theta = theta - v
        trajectory.append(theta.copy())
    return np.array(trajectory)


def run_adam(start, lr, beta1, beta2, n_steps, epsilon=1e-8):
    """Adam optimizer with bias correction."""
    theta = np.array(start, dtype=float)
    m = np.zeros(2)
    v = np.zeros(2)
    trajectory = [theta.copy()]
    for t in range(1, n_steps + 1):
        g = grad_beale(theta[0], theta[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        trajectory.append(theta.copy())
    return np.array(trajectory)


# ── Run optimizers ───────────────────────────────────────────────────────────

START = np.array([0.5, 3.5])
N_STEPS = 200

traj_sgd = run_sgd(START, lr=0.0001, n_steps=N_STEPS)
traj_mom = run_momentum(START, lr=0.0001, gamma=0.9, n_steps=N_STEPS)
traj_adam = run_adam(START, lr=0.01, beta1=0.9, beta2=0.999, n_steps=N_STEPS)

# Compute loss histories
loss_sgd = np.array([beale(p[0], p[1]) for p in traj_sgd])
loss_mom = np.array([beale(p[0], p[1]) for p in traj_mom])
loss_adam = np.array([beale(p[0], p[1]) for p in traj_adam])

# ── Colors ───────────────────────────────────────────────────────────────────

C_SGD = COLORS["blue"]
C_MOM = COLORS["orange"]
C_ADAM = COLORS["green"]

# ── Figure ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 11.1.3　优化器对比与训练过程",
             fontsize=22, fontweight="bold", y=0.98)

# ═════════════════════════════════════════════════════════════════════════════
# Panel (a): Optimizer trajectories on Beale function contour
# ═════════════════════════════════════════════════════════════════════════════

x_grid = np.linspace(-1.0, 4.5, 400)
y_grid = np.linspace(-1.0, 4.5, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = beale(X, Y)

# Use log-spaced contour levels for better visualization
levels = np.logspace(-1, 5, 30)
ax1.contourf(X, Y, Z, levels=levels, cmap="RdYlBu_r", alpha=0.75)
ax1.contour(X, Y, Z, levels=levels, colors="gray", linewidths=0.4, alpha=0.5)

# Mark the global minimum
ax1.plot(3.0, 0.5, marker="P", color="black", markersize=12, zorder=10,
         markeredgecolor="white", markeredgewidth=1.2, label="最优点 (3, 0.5)")

MARKER_STEP = 20  # place a dot every N steps

# Helper to plot a single trajectory
def plot_trajectory(ax, traj, color, label):
    ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.0, alpha=0.85,
            label=label, zorder=5)
    # Markers at regular intervals
    idx = np.arange(0, len(traj), MARKER_STEP)
    ax.plot(traj[idx, 0], traj[idx, 1], 'o', color=color, markersize=4,
            zorder=6, alpha=0.7)
    # Start marker
    ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, zorder=7)
    # End marker
    ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, zorder=7)


plot_trajectory(ax1, traj_sgd, C_SGD, "SGD")
plot_trajectory(ax1, traj_mom, C_MOM, "Momentum")
plot_trajectory(ax1, traj_adam, C_ADAM, "Adam")

ax1.set_xlabel("$x_1$", fontsize=14)
ax1.set_ylabel("$x_2$", fontsize=14)
ax1.set_title("(a) 优化器轨迹", fontsize=17, fontweight="bold")
ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)
ax1.set_xlim(-1.0, 4.5)
ax1.set_ylim(-1.0, 4.5)
# Restore all spines for the contour plot
for spine in ax1.spines.values():
    spine.set_visible(True)
ax1.grid(False)

# ═════════════════════════════════════════════════════════════════════════════
# Panel (b): Loss vs iteration
# ═════════════════════════════════════════════════════════════════════════════

steps = np.arange(N_STEPS + 1)
ax2.semilogy(steps, loss_sgd, color=C_SGD, lw=2.0, label="SGD")
ax2.semilogy(steps, loss_mom, color=C_MOM, lw=2.0, label="Momentum")
ax2.semilogy(steps, loss_adam, color=C_ADAM, lw=2.0, label="Adam")

ax2.set_xlabel("迭代步数", fontsize=14)
ax2.set_ylabel("损失值", fontsize=14)
ax2.set_title("(b) 损失值收敛曲线", fontsize=17, fontweight="bold")
ax2.legend(loc="upper right", fontsize=11, framealpha=0.9)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, __file__, "fig11_1_03_training_optimization")
