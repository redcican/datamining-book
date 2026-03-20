"""fig11_6_02_dqn_architecture.py
(a) DQN 网络架构  (b) 经验回放与目标网络"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5),
                                gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("图 11.6.2　DQN 架构与核心技术",
             fontsize=22, fontweight="bold", y=0.98)

# ── Helper functions ─────────────────────────────────────────────
LW = 2.0

def draw_box(ax, cx, cy, w, h, text, color, fontsize=16,
             text_color="white", linestyle="-", linewidth=1.8, alpha=0.9):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black",
        linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)

def draw_box_outline(ax, cx, cy, w, h, text, color, fontsize=16,
                     text_color=None, linestyle="-", linewidth=1.8):
    tc = text_color if text_color else color
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor="white", edgecolor=color,
        linewidth=linewidth, linestyle=linestyle, alpha=0.95)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=tc)

def arr(ax, x1, y1, x2, y2, color=COLORS["gray"], lw=LW,
        linestyle="-", mutation_scale=14):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, linestyle=linestyle,
                                mutation_scale=mutation_scale))

# ══════════════════════════════════════════════════════════════════
# (a) DQN 网络架构
# ══════════════════════════════════════════════════════════════════
ax1.set_axis_off()
ax1.set_xlim(-1, 23)
ax1.set_ylim(-1.5, 13)
ax1.set_aspect("equal")

c_input = COLORS["blue"]
c_hidden = COLORS["teal"]
c_output = COLORS["purple"]
c_argmax = COLORS["orange"]
c_line = COLORS["gray"]

# -- Input box --
draw_box(ax1, 2.0, 6.0, 2.2, 3.5, "", c_input, fontsize=16)
ax1.text(2.0, 6.5, "状态", fontsize=15, fontweight="bold",
         color="white", ha="center", va="center")
ax1.text(2.0, 5.5, "$s$", fontsize=18, fontweight="bold",
         color="white", ha="center", va="center")
ax1.text(2.0, 3.5, "输入层", fontsize=14, fontweight="bold",
         color=c_input, ha="center")

# Dimension annotation: input → hidden1
ax1.text(5.5, 10.5, "$n$", fontsize=14, fontweight="bold",
         color=c_line, ha="center",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c_line, alpha=0.7))

# -- Hidden layer 1 --
draw_box(ax1, 7.5, 6.0, 2.2, 4.5, "", c_hidden, fontsize=14)
ax1.text(7.5, 6.5, "FC 128", fontsize=14, fontweight="bold",
         color="white", ha="center", va="center")
ax1.text(7.5, 5.3, "ReLU", fontsize=13, fontweight="bold",
         color="white", ha="center", va="center")
ax1.text(7.5, 3.0, "隐藏层 1", fontsize=14, fontweight="bold",
         color=c_hidden, ha="center")

arr(ax1, 3.1, 6.0, 6.4, 6.0, color=c_line, lw=2.5)

# Dimension annotation: hidden1 → hidden2
ax1.text(10.5, 10.5, "128", fontsize=14, fontweight="bold",
         color=c_line, ha="center",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c_line, alpha=0.7))

# -- Hidden layer 2 --
draw_box(ax1, 13.0, 6.0, 2.2, 4.5, "", c_hidden, fontsize=14)
ax1.text(13.0, 6.5, "FC 128", fontsize=14, fontweight="bold",
         color="white", ha="center", va="center")
ax1.text(13.0, 5.3, "ReLU", fontsize=13, fontweight="bold",
         color="white", ha="center", va="center")
ax1.text(13.0, 3.0, "隐藏层 2", fontsize=14, fontweight="bold",
         color=c_hidden, ha="center")

arr(ax1, 8.6, 6.0, 11.9, 6.0, color=c_line, lw=2.5)

# Dimension annotation: hidden2 → output
ax1.text(15.7, 10.5, "128", fontsize=14, fontweight="bold",
         color=c_line, ha="center",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c_line, alpha=0.7))

# -- Output Q values (multiple nodes) --
q_labels = ["$Q(s,a_1)$", "$Q(s,a_2)$", "$Q(s,a_3)$", "$Q(s,a_4)$"]
q_ys = [9.5, 7.5, 5.5, 3.5]

for ql, qy in zip(q_labels, q_ys):
    draw_box(ax1, 18.0, qy, 2.5, 1.3, ql, c_output, fontsize=13)

# Arrows from hidden2 to each Q output
for qy in q_ys:
    arr(ax1, 14.1, 6.0 + (qy - 6.0) * 0.3, 16.75, qy,
        color=c_line, lw=1.8)

ax1.text(18.0, 1.5, "输出层", fontsize=14, fontweight="bold",
         color=c_output, ha="center")

# -- argmax box --
draw_box(ax1, 21.5, 6.5, 1.8, 1.8, "argmax", c_argmax, fontsize=13)

# Arrows from Q outputs to argmax
for qy in q_ys:
    arr(ax1, 19.25, qy, 20.6, 6.5 + (qy - 6.5) * 0.1,
        color=c_output, lw=1.5)

# Output action label
ax1.text(21.5, 4.0, "动作 $a$", fontsize=16, fontweight="bold",
         color=c_argmax, ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                   ec=c_argmax, alpha=0.9))
arr(ax1, 21.5, 5.6, 21.5, 4.7, color=c_argmax, lw=2.5)

# Architecture formula
ax1.text(11.0, 12.0,
         "$a^* = \\arg\\max_{a}\\, Q(s, a;\\theta)$",
         fontsize=15, fontweight="bold", color=c_line, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=c_line, alpha=0.9))

ax1.set_title("(a) DQN 网络架构", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 经验回放与目标网络
# ══════════════════════════════════════════════════════════════════
ax2.set_axis_off()
ax2.set_xlim(-1, 21)
ax2.set_ylim(-2, 16)
ax2.set_aspect("equal")

c_env = COLORS["green"]
c_agent = COLORS["blue"]
c_buffer = COLORS["orange"]
c_online = COLORS["teal"]
c_target = COLORS["purple"]
c_loss = COLORS["red"]

# -- Environment box (top-left) --
draw_box(ax2, 4.0, 14.0, 3.5, 1.8, "环境", c_env, fontsize=16)

# -- Agent box (top-right) --
draw_box(ax2, 16.0, 14.0, 3.5, 1.8, "智能体", c_agent, fontsize=16)

# Interaction arrows: env → agent (state, reward) and agent → env (action)
arr(ax2, 5.75, 14.6, 14.25, 14.6, color=c_env, lw=2.0)
ax2.text(10.0, 15.2, "状态 $s$, 奖励 $r$", fontsize=14, fontweight="bold",
         color=c_env, ha="center")

arr(ax2, 14.25, 13.4, 5.75, 13.4, color=c_agent, lw=2.0)
ax2.text(10.0, 12.6, "动作 $a$", fontsize=14, fontweight="bold",
         color=c_agent, ha="center")

# -- Experience buffer (middle) --
buf_cx, buf_cy, buf_w, buf_h = 10.0, 8.5, 8.0, 2.5
draw_box_outline(ax2, buf_cx, buf_cy, buf_w, buf_h,
                 "", c_buffer, fontsize=14, linewidth=2.2)
ax2.text(buf_cx, 9.2, "经验缓冲区 $\\mathcal{D}$",
         fontsize=16, fontweight="bold", color=c_buffer, ha="center")
# Stored tuples
ax2.text(buf_cx, 7.8,
         "$(s_t,\\, a_t,\\, r_t,\\, s_{t+1})$",
         fontsize=15, fontweight="bold", color=c_buffer, ha="center",
         alpha=0.85)

# Arrow: interaction → buffer (store)
arr(ax2, 10.0, 12.4, 10.0, 9.8, color=c_buffer, lw=2.2)
ax2.text(8.5, 11.2, "存储", fontsize=14, fontweight="bold",
         color=c_buffer, ha="center",
         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                   ec=c_buffer, alpha=0.8))

# -- Online network Q_θ (bottom-left) --
draw_box(ax2, 5.0, 3.5, 4.0, 2.0, "", c_online, fontsize=15)
ax2.text(5.0, 4.0, "在线网络", fontsize=14, fontweight="bold",
         color="white", ha="center", va="center")
ax2.text(5.0, 2.9, "$Q_\\theta$", fontsize=16, fontweight="bold",
         color="white", ha="center", va="center")

# Arrow: buffer → online (random sample)
arr(ax2, 8.0, 7.2, 5.5, 4.6, color=c_online, lw=2.2)
ax2.text(5.0, 6.2, "随机采样\nmini-batch", fontsize=13, fontweight="bold",
         color=c_online, ha="center",
         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                   ec=c_online, alpha=0.8))

# -- Target network Q_θ⁻ (bottom-right, dashed border) --
draw_box_outline(ax2, 15.5, 3.5, 4.0, 2.0,
                 "", c_target, fontsize=15,
                 linestyle="--", linewidth=2.2)
ax2.text(15.5, 4.0, "目标网络", fontsize=14, fontweight="bold",
         color=c_target, ha="center", va="center")
ax2.text(15.5, 2.9, "$Q_{\\theta^-}$", fontsize=16, fontweight="bold",
         color=c_target, ha="center", va="center")

# Arrow: online → target (sync every C steps, dashed)
arr(ax2, 7.0, 3.5, 13.5, 3.5, color=c_target, lw=2.0, linestyle="--")
ax2.text(10.2, 2.5, "每 $C$ 步同步", fontsize=14, fontweight="bold",
         color=c_target, ha="center",
         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                   ec=c_target, alpha=0.8))

# -- Loss computation --
ax2.text(10.2, 0.2,
         "$L(\\theta) = \\mathbb{E}\\left["
         "(r + \\gamma \\max_{a\'} Q_{\\theta^-}(s\', a\') "
         "- Q_\\theta(s,a))^2\\right]$",
         fontsize=13, fontweight="bold", color=c_loss, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=c_loss, alpha=0.9))

# Arrows from online and target to loss
arr(ax2, 5.0, 2.5, 7.5, 0.6, color=c_online, lw=1.8)
arr(ax2, 15.5, 2.5, 13.0, 0.6, color=c_target, lw=1.8)

# Gradient feedback arrow (loss → online, dashed)
arr(ax2, 7.0, -0.2, 4.5, 2.4, color=c_loss, lw=1.8, linestyle="--")
ax2.text(3.8, 0.8, "$\\nabla_\\theta L$", fontsize=14, fontweight="bold",
         color=c_loss, ha="center")

ax2.set_title("(b) 经验回放与目标网络", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_6_02_dqn_architecture")
