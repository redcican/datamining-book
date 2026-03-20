"""
fig9_1_03_word_distribution.py
词频 Zipf 分布 + 词云可视化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 模拟 Zipf 词频分布 ───────────────────────────────────────────
n_vocab = 8000
ranks = np.arange(1, n_vocab + 1)
alpha = 1.05
freqs = (ranks ** (-alpha))
total_tokens = 120000
freqs = freqs / freqs.sum() * total_tokens
noise = np.random.lognormal(0, 0.15, n_vocab)
freqs = freqs * noise
freqs = np.sort(freqs)[::-1]
# ── 各类别特征词数据（用于词云）─────────────────────────────────
word_freq_data = {
    "比赛": 850, "冠军": 720, "进球": 650, "联赛": 580, "教练": 510,
    "球员": 480, "赛季": 460, "足球": 440, "得分": 420, "胜利": 400,
    "技术": 920, "发布": 780, "芯片": 680, "智能": 550, "研发": 480,
    "数据": 750, "算法": 620, "模型": 580, "网络": 520, "平台": 460,
    "市场": 880, "增长": 750, "投资": 620, "经济": 560, "股市": 450,
    "基金": 420, "利率": 380, "资本": 360, "金融": 340, "贸易": 320,
    "演员": 780, "电影": 700, "票房": 630, "综艺": 520, "导演": 440,
    "明星": 460, "节目": 380, "观众": 350, "表演": 330, "制作": 310,
    "学校": 820, "考试": 680, "教学": 600, "课程": 530, "招生": 460,
    "学生": 720, "高考": 580, "培训": 440, "教育": 650, "老师": 520,
}
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.1.3　文本预处理结果分析",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：Zipf 词频 log-log 图 ─────────────────────────────────
ax = axes[0]
ax.scatter(ranks, freqs, s=4, alpha=0.4, color=COLORS["blue"])
log_r = np.log10(ranks[:5000])
log_f = np.log10(freqs[:5000])
coeffs = np.polyfit(log_r, log_f, 1)
fit_line = 10 ** (coeffs[0] * np.log10(ranks) + coeffs[1])
ax.plot(ranks, fit_line, "--", color=COLORS["red"], lw=2,
        label=f"线性拟合 (斜率={coeffs[0]:.2f})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("词频排名 r (log)", fontsize=16)
ax.set_ylabel("词频 f(r) (log)", fontsize=16)
ax.set_title("(a) Zipf 词频分布", fontsize=17)
ax.legend(fontsize=14, loc="upper right")
ax.tick_params(labelsize=14)
ax.annotate("高频词区\n(功能词+通用词)",
            xy=(5, freqs[4]), xytext=(30, freqs[4] * 0.6),
            fontsize=12, color=COLORS["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))
ax.annotate("低频长尾\n(专有名词、罕见词)",
            xy=(5000, freqs[4999]), xytext=(800, freqs[4999] * 0.1),
            fontsize=12, color=COLORS["orange"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.5))
# ── 右面板：词云可视化 ────────────────────────────────────────────
ax = axes[1]
# 查找中文字体
import matplotlib.font_manager as fm
zh_fonts = [f.fname for f in fm.fontManager.ttflist
            if any(k in f.name for k in ["SimHei", "Heiti", "Noto Sans CJK",
                                          "WenQuanYi", "Source Han Sans",
                                          "Microsoft YaHei", "PingFang"])]
font_path = zh_fonts[0] if zh_fonts else None
# 自定义颜色函数：按类别分色
category_words = {
    "体育": {"比赛", "冠军", "进球", "联赛", "教练", "球员", "赛季", "足球", "得分", "胜利"},
    "科技": {"技术", "发布", "芯片", "智能", "研发", "数据", "算法", "模型", "网络", "平台"},
    "财经": {"市场", "增长", "投资", "经济", "股市", "基金", "利率", "资本", "金融", "贸易"},
    "娱乐": {"演员", "电影", "票房", "综艺", "导演", "明星", "节目", "观众", "表演", "制作"},
    "教育": {"学校", "考试", "教学", "课程", "招生", "学生", "高考", "培训", "教育", "老师"},
}
cat_color_map = {
    "体育": COLORS["blue"], "科技": COLORS["red"], "财经": COLORS["green"],
    "娱乐": COLORS["orange"], "教育": COLORS["purple"],
}
def word_to_color(word):
    for cat, words in category_words.items():
        if word in words:
            return cat_color_map[cat]
    return COLORS["gray"]
def color_func(word, **kwargs):
    c = word_to_color(word)
    # 转换 hex 到 RGB
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return f"rgb({r},{g},{b})"
wc = WordCloud(
    width=800, height=500, background_color="white",
    font_path=font_path, max_words=50, max_font_size=90,
    min_font_size=14, prefer_horizontal=0.8,
    color_func=color_func, random_state=42,
)
wc.generate_from_frequencies(word_freq_data)
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("(b) 类别特征词词云", fontsize=17)
# 手动图例
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=c, label=cat)
                  for cat, c in cat_color_map.items()]
ax.legend(handles=legend_patches, fontsize=12, loc="lower right",
          framealpha=0.9)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_1_03_word_distribution")
