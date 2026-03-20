"""fig11_6_01_rl_framework.py
(a) 智能体-环境交互循环  (b) 网格世界值函数"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5),
                                gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("图 11.6.1　强化学习基本框架与 MDP",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) 智能体-环境交互循环
# ══════════════════════════════════════════════════════════════════
ax = ax1
ax.set_axis_off()
ax.set_xlim(-1, 21)
ax.set_ylim(-2, 14)
ax.set_aspect("equal")

c_agent = COLORS["blue"]
c_env = COLORS["green"]
c_policy = COLORS["teal"]
c_action = COLORS["red"]
c_state = COLORS["purple"]
c_reward = COLORS["orange"]
c_line = COLORS["gray"]
LW = 2.0


def draw_box(ax, cx, cy, w, h, text, color, fontsize=16,
             text_color="white", alpha=0.9):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.2",
        facecolor=color, edgecolor="black", linewidth=1.8, alpha=alpha)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def arr(ax, x1, y1, x2, y2, color=c_line, lw=LW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16))


# ── Agent box (top center) ──
agent_cx, agent_cy = 10.0, 10.5
draw_box(ax, agent_cx, agent_cy, 6.0, 2.5,
         "智能体 Agent", c_agent, fontsize=17)

# ── Policy box (inside/near agent) ──
policy_cx, policy_cy = 10.0, 8.3
draw_box(ax, policy_cx, policy_cy, 3.0, 1.2,
         "策略 $\\pi(a|s)$", c_policy, fontsize=14, alpha=0.95)

# ── Environment box (bottom center) ──
env_cx, env_cy = 10.0, 2.5
draw_box(ax, env_cx, env_cy, 6.0, 2.5,
         "环境 Environment", c_env, fontsize=17)

# ── Arrow: Agent → Environment (right side, downward) — action ──
act_x = 16.0
arr(ax, agent_cx + 3.0, agent_cy - 0.5, act_x, agent_cy - 0.5,
    color=c_action, lw=2.5)
arr(ax, act_x, agent_cy - 0.5, act_x, env_cy + 0.5,
    color=c_action, lw=2.5)
arr(ax, act_x, env_cy + 0.5, env_cx + 3.0, env_cy + 0.5,
    color=c_action, lw=2.5)

# Action label
ax.text(act_x + 0.5, (agent_cy + env_cy) / 2 + 1.2, "$a_t$",
        fontsize=17, fontweight="bold", color=c_action, ha="center")
ax.text(act_x + 0.5, (agent_cy + env_cy) / 2 + 0.0, "动作",
        fontsize=15, fontweight="bold", color=c_action, ha="center")

# ── Arrow: Environment → Agent (left side, upward) — state & reward ──
ret_x = 4.0
arr(ax, env_cx - 3.0, env_cy + 0.5, ret_x, env_cy + 0.5,
    color=c_state, lw=2.5)
arr(ax, ret_x, env_cy + 0.5, ret_x, agent_cy - 0.5,
    color=c_state, lw=2.5)
arr(ax, ret_x, agent_cy - 0.5, agent_cx - 3.0, agent_cy - 0.5,
    color=c_state, lw=2.5)

# State label
ax.text(ret_x - 1.2, (agent_cy + env_cy) / 2 + 1.2, "$s_{t+1}$",
        fontsize=17, fontweight="bold", color=c_state, ha="center")
ax.text(ret_x - 1.2, (agent_cy + env_cy) / 2 + 0.0, "状态",
        fontsize=15, fontweight="bold", color=c_state, ha="center")

# Reward label (slightly below state)
ax.text(ret_x - 1.2, (agent_cy + env_cy) / 2 - 1.3, "$r_t$",
        fontsize=17, fontweight="bold", color=c_reward, ha="center")
ax.text(ret_x - 1.2, (agent_cy + env_cy) / 2 - 2.5, "奖励",
        fontsize=15, fontweight="bold", color=c_reward, ha="center")

# ── MDP tuple annotation ──
ax.text(10.0, -1.0,
        "MDP: $\\langle \\mathcal{S},\\, \\mathcal{A},\\,"
        " P(s'|s,a),\\, R(s,a),\\, \\gamma \\rangle$",
        fontsize=15, fontweight="bold", color=c_line, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=c_line, alpha=0.9))

ax.set_title("(a) 智能体-环境交互", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 网格世界 V*(s) — 值迭代
# ══════════════════════════════════════════════════════════════════
ax = ax2
grid_size = 5
gamma = 0.9
reward_step = -0.04
goal = (4, 4)
trap = (1, 3)

# Value iteration
V = np.zeros((grid_size, grid_size))
V[goal] = 1.0
V[trap] = -1.0

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

for _ in range(200):
    V_new = V.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal or (i, j) == trap:
                continue
            q_values = []
            for di, dj in actions:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    q_values.append(reward_step + gamma * V[ni, nj])
                else:
                    q_values.append(reward_step + gamma * V[i, j])
            V_new[i, j] = max(q_values)
    V = V_new

# Display: imshow expects (row, col) where row 0 is top
# We want row 0 to be top of grid
V_display = V.copy()

im = ax.imshow(V_display, cmap="RdYlGn", origin="upper",
               vmin=-1.0, vmax=1.0)

# Annotate cell values
for i in range(grid_size):
    for j in range(grid_size):
        val = V_display[i, j]
        text_color = "white" if abs(val) > 0.6 else "black"
        fontw = "bold" if (i, j) == goal or (i, j) == trap else "normal"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=14, fontweight=fontw, color=text_color)

# Mark goal
ax.plot(goal[1], goal[0], marker="*", markersize=22,
        color="gold", markeredgecolor="black", markeredgewidth=1.5,
        zorder=5)
ax.text(goal[1], goal[0] + 0.38, "目标", fontsize=11,
        ha="center", va="top", color="black", fontweight="bold")

# Mark trap
ax.plot(trap[1], trap[0], marker="X", markersize=18,
        color=COLORS["red"], markeredgecolor="black", markeredgewidth=1.5,
        zorder=5)
ax.text(trap[1], trap[0] + 0.38, "陷阱", fontsize=11,
        ha="center", va="top", color="black", fontweight="bold")

# Axis labels
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.set_xticklabels(range(grid_size), fontsize=14)
ax.set_yticklabels(range(grid_size), fontsize=14)
ax.set_xlabel("列", fontsize=16)
ax.set_ylabel("行", fontsize=16)

# Grid lines
for edge in range(grid_size + 1):
    ax.axhline(edge - 0.5, color="black", lw=0.8, alpha=0.5)
    ax.axvline(edge - 0.5, color="black", lw=0.8, alpha=0.5)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("$V^*(s)$", fontsize=16, fontweight="bold")
cbar.ax.tick_params(labelsize=14)

# Annotation: gamma value
ax.text(grid_size - 0.5, grid_size + 0.1,
        f"$\\gamma = {gamma}$", fontsize=14,
        ha="right", va="top", color=c_line, fontweight="bold")

ax.set_title("(b) 网格世界 $V^*(s)$", fontsize=17, fontweight="bold", pad=15)

# Re-enable spines for heatmap
for spine in ax.spines.values():
    spine.set_visible(True)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_6_01_rl_framework")
