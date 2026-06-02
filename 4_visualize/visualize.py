"""
4_visualize/visualize.py
─────────────────────────
모든 분석 결과를 시각화합니다.

생성 그래프:
  [1] 연도별 기술 키워드 등장 추이 (라인 차트)
  [2] 기술 공출현 히트맵
  [3] 공출현 네트워크 그래프
  [4] BERTopic 토픽 비중 (바 차트)
  [5] 임베딩 UMAP 산점도
  [6] OCR ↔ 기술 유사도 수평 바 차트

실행: python 4_visualize/visualize.py
출력: outputs/fig1_tech_trend.png
      outputs/fig2_cooccurrence_heatmap.png
      outputs/fig3_cooccurrence_network.png
      outputs/fig4_topic_distribution.png
      outputs/fig5_umap_scatter.png
      outputs/fig6_similarity_bar.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import networkx as nx
from config import PROC_DIR, OUTPUT_DIR, TECH_KEYWORDS
import platform

# ── 폰트 설정 (한글 지원) ────────────────────────────
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

PALETTE = sns.color_palette("tab20", n_colors=20)




# ─────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────

def save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")


def load_if_exists(filename: str) -> pd.DataFrame | None:
    path = os.path.join(PROC_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    print(f"  ⚠ 파일 없음: {path}")
    return None


# ─────────────────────────────────────────────────────
# [1-1] 연도별 핵심 기술 추이
# ─────────────────────────────────────────────────────

MODEL_TECHS = [
    "Deep Learning",
    "Transformer / Attention",
    "LSTM / RNN",
]

APPLICATION_TECHS = [
    "Large Language Model",
    "Scene Text",
    "Handwriting",
    "Document Analysis",
    "Multilingual",
    "Historical Documents",
    "Medical / Healthcare",
    "Edge / Mobile",
]


def fig1_model_technology_trend():
    df = load_if_exists("tech_trend_yearly.csv")
    if df is None:
        return

    techs = [t for t in MODEL_TECHS if t in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, tech in enumerate(techs):
        ax.plot(
            df["year"],
            df[tech],
            marker="o",
            label=tech,
            color=PALETTE[i % len(PALETTE)],
            linewidth=2.2,
        )

    ax.set_title("Core OCR Technology Trends (2016–2026)", fontsize=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Paper Count")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    save(fig, "fig1_model_technology_trend.png")


# ─────────────────────────────────────────────────────
# [1-2] 연도별 OCR 활용 분야 추이
# ─────────────────────────────────────────────────────

def fig2_application_trend():
    df = load_if_exists("tech_trend_yearly.csv")
    if df is None:
        return

    techs = [t for t in APPLICATION_TECHS if t in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, tech in enumerate(techs):
        ax.plot(
            df["year"],
            df[tech],
            marker="o",
            label=tech,
            color=PALETTE[(i + 4) % len(PALETTE)],
            linewidth=2.0,
        )

    ax.set_title("OCR Application Area Trends (2016–2026)", fontsize=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Paper Count")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    save(fig, "fig2_application_trend.png")


# ─────────────────────────────────────────────────────
# [2] 공출현 히트맵
# ─────────────────────────────────────────────────────

def fig2_cooccurrence_heatmap():
    df = load_if_exists("cooccurrence_matrix.csv")
    if df is None:
        return

    df = df.set_index(df.columns[0]) if df.columns[0] not in TECH_KEYWORDS else df
    mat = df.values.astype(float)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        mat,
        xticklabels=df.columns,
        yticklabels=df.index,
        annot=True, fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("기술 키워드 공출현 매트릭스", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    save(fig, "fig2_cooccurrence_heatmap.png")


# ─────────────────────────────────────────────────────
# [3] 공출현 네트워크 그래프
# ─────────────────────────────────────────────────────

def fig3_cooccurrence_network():
    df = load_if_exists("cooccurrence_matrix.csv")
    if df is None:
        return

    df = df.set_index(df.columns[0]) if df.columns[0] not in TECH_KEYWORDS else df
    techs = list(df.index)

    G = nx.Graph()
    G.add_nodes_from(techs)

    threshold = df.values.max() * 0.05   # 최대값의 5% 이상만 엣지로
    for i, t1 in enumerate(techs):
        for j, t2 in enumerate(techs):
            if i < j:
                val = float(df.loc[t1, t2]) if t1 in df.index and t2 in df.columns else 0
                if val >= threshold:
                    G.add_edge(t1, t2, weight=val)

    pos = nx.spring_layout(G, seed=42, k=2)
    node_sizes = [800 + df.loc[t].sum() * 0.5 for t in techs if t in df.index]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=PALETTE[:len(G.nodes())], alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        width=[w / max_w * 6 for w in edge_weights],
        edge_color=edge_weights, edge_cmap=plt.cm.Blues,
        alpha=0.6, ax=ax,
    )
    ax.set_title("OCR 기술 공출현 네트워크", fontsize=15)
    ax.axis("off")
    save(fig, "fig3_cooccurrence_network.png")


# ─────────────────────────────────────────────────────
# [4] BERTopic 토픽 분포
# ─────────────────────────────────────────────────────

def fig4_topic_distribution():
    df = load_if_exists("topic_info.csv")
    if df is None:
        return

    df = df[df["Topic"] != -1].head(20).sort_values("Count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(df["Name"], df["Count"], color=sns.color_palette("viridis", len(df)))
    ax.set_xlabel("논문 수")
    ax.set_title("BERTopic 상위 토픽 분포", fontsize=14)
    for bar, val in zip(bars, df["Count"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=7)
    save(fig, "fig4_topic_distribution.png")


# ─────────────────────────────────────────────────────
# [5] UMAP 산점도
# ─────────────────────────────────────────────────────

def fig5_umap_scatter():
    df = load_if_exists("umap_coords.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(13, 11))

    # 기술 태그별 색상 매핑
    all_techs = list(TECH_KEYWORDS.keys())
    color_map  = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(all_techs)}

    plotted = set()
    if "tech_tags" in df.columns:
        for tech in all_techs:
            mask = df["tech_tags"].str.contains(tech, na=False, regex=False)
            sub  = df[mask]
            if len(sub) == 0:
                continue
            ax.scatter(sub["umap_x"], sub["umap_y"],
                       c=[color_map[tech]], s=5, alpha=0.5,
                       label=f"{tech} ({len(sub)})", rasterized=True)
            plotted.update(sub.index)

        # 미분류
        unlabeled = df[~df.index.isin(plotted)]
        if len(unlabeled):
            ax.scatter(unlabeled["umap_x"], unlabeled["umap_y"],
                       c="lightgray", s=3, alpha=0.3, label="기타")
    else:
        ax.scatter(df["umap_x"], df["umap_y"], s=3, alpha=0.4, c="steelblue")

    ax.set_title("OCR 논문 임베딩 UMAP 시각화 (기술 분야별)", fontsize=14)
    ax.legend(loc="upper right", fontsize=6, markerscale=3, ncol=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    save(fig, "fig5_umap_scatter.png")


# ─────────────────────────────────────────────────────
# [6] OCR ↔ 기술 유사도 바 차트
# ─────────────────────────────────────────────────────

def fig6_similarity_bar():
    df = load_if_exists("similarity_scores.csv")
    if df is None:
        return

    df = df.sort_values("cos_similarity_to_ocr", ascending=True)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["technology"], df["cos_similarity_to_ocr"], color=colors)
    ax.set_xlabel("Cosine Similarity (OCR 논문과의 의미적 유사도)")
    ax.set_title("임베딩 기반 OCR ↔ 기술 분야 의미적 유사도", fontsize=14)
    ax.axvline(df["cos_similarity_to_ocr"].mean(), color="red",
               linestyle="--", linewidth=1.2, label="평균")
    ax.legend(fontsize=9)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["cos_similarity_to_ocr"] + 0.002, i,
                f'{row["cos_similarity_to_ocr"]:.3f}', va="center", fontsize=8)
    save(fig, "fig6_similarity_bar.png")


# ─────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" 통합 시각화")
    print("=" * 60)

    fig1_model_technology_trend()
    fig2_application_trend()

    fig2_cooccurrence_heatmap()
    fig3_cooccurrence_network()
    fig4_topic_distribution()
    fig5_umap_scatter()
    fig6_similarity_bar()

    print(f"\n 모든 시각화 완료 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
