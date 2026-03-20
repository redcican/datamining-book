"""
fig9_2_03_representation_comparison.py
三种文本表示方法的相似度矩阵对比（seaborn 热力图）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from shared.plot_config import apply_style, save_fig, COLORS
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 模拟文档数据 ──────────────────────────────────────────────────
docs = [
    "football match championship goal victory",
    "soccer game tournament score winning",
    "basketball player season team league",
    "artificial intelligence machine learning model",
    "deep learning neural network training",
    "AI algorithm data prediction accuracy",
    "football player transferred to new team",
    "machine learning model prediction results",
]
doc_short = ["体育1", "体育2", "体育3", "科技1", "科技2", "科技3", "体育4", "科技4"]
# ── 构建词袋 ──────────────────────────────────────────────────────
vocab = sorted(set(w for d in docs for w in d.split()))
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
bow_matrix = np.zeros((len(docs), V))
for i, d in enumerate(docs):
    for w in d.split():
        bow_matrix[i, word2idx[w]] += 1
# ── TF-IDF ────────────────────────────────────────────────────────
N = len(docs)
df = np.sum(bow_matrix > 0, axis=0)
idf = np.log(N / (df + 1e-8))
tfidf_matrix = bow_matrix * idf
# ── 模拟词向量 ────────────────────────────────────────────────────
dim = 50
word_vecs = {}
sport_words = {"football", "soccer", "match", "game", "championship",
               "tournament", "goal", "score", "victory", "winning",
               "basketball", "player", "season", "team", "league",
               "transferred", "new", "to"}
tech_words = {"artificial", "intelligence", "machine", "learning",
              "model", "deep", "neural", "network", "training",
              "AI", "algorithm", "data", "prediction", "accuracy",
              "results"}
for w in vocab:
    base = np.random.randn(dim) * 0.08
    if w in sport_words:
        base[:25] += np.random.randn(25) * 0.2 + 0.5
    elif w in tech_words:
        base[25:] += np.random.randn(25) * 0.2 + 0.5
    word_vecs[w] = base
embed_matrix = np.zeros((len(docs), dim))
for i, d in enumerate(docs):
    words = d.split()
    embed_matrix[i] = np.mean([word_vecs[w] for w in words if w in word_vecs],
                              axis=0)
# ── 余弦相似度 ────────────────────────────────────────────────────
def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_norm = X / norms
    return X_norm @ X_norm.T
sim_bow = cosine_sim_matrix(bow_matrix)
sim_tfidf = cosine_sim_matrix(tfidf_matrix)
sim_embed = cosine_sim_matrix(embed_matrix)
# ── 绘图（seaborn 热力图）────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("图 9.2.3　三种文本表示方法的相似度矩阵对比",
             fontsize=20, fontweight="bold", y=1.03)
titles = ["(a) BoW 余弦相似度", "(b) TF-IDF 余弦相似度", "(c) 平均词向量余弦相似度"]
matrices = [sim_bow, sim_tfidf, sim_embed]
cmaps = ["Blues", "Oranges", "Purples"]
for ax, title, sim, cmap in zip(axes, titles, matrices, cmaps):
    sim_df = pd.DataFrame(sim, index=doc_short, columns=doc_short)
    sns.heatmap(sim_df, ax=ax, cmap=cmap, vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8.5},
                linewidths=0.5, linecolor="white", square=True,
                cbar_kws={"shrink": 0.75})
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=10.5)
    # 画类别分隔线
    for pos in [3, 6]:
        ax.axhline(pos, color=COLORS["red"], lw=1.5, ls="--", alpha=0.5)
        ax.axvline(pos, color=COLORS["red"], lw=1.5, ls="--", alpha=0.5)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_2_03_representation_comparison")
