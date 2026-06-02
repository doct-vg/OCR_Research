"""
3_analyze/bertopic_model.py

BERTopic 토픽 모델링
- 초록 임베딩 → UMAP 차원 축소 → HDBSCAN 클러스터링
- 자동 토픽 레이블링
- 연도별 토픽 비중 변화

실행:
python 3_analyze/bertopic_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from config import PROC_DIR, OUTPUT_DIR, BERTOPIC_N_TOPICS, EMBED_MODEL


def run_bertopic(df: pd.DataFrame):
    abstracts = df["abstract_clean"].fillna("").astype(str).tolist()
    timestamps = df["published"].fillna("2020-01-01").astype(str).tolist()

    print(f"  초록 수: {len(abstracts):,}")
    print(f"  임베딩 모델: {EMBED_MODEL}")

    embedding_model = SentenceTransformer(EMBED_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=BERTOPIC_N_TOPICS,
        verbose=True,
        calculate_probabilities=True,
    )

    print("\n  BERTopic 학습 중... (시간이 걸릴 수 있습니다)")
    topics, probs = topic_model.fit_transform(abstracts)

    # 1) 논문별 토픽 할당
    df["topic_id"] = topics

    # probs가 None일 수도 있어서 안전 처리
    if probs is not None:
        df["topic_prob"] = [
            max(p) if hasattr(p, "__len__") else p
            for p in probs
        ]
    else:
        df["topic_prob"] = None

    topic_info = topic_model.get_topic_info()
    topic_label_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_label"] = df["topic_id"].map(topic_label_map)

    assign_out = os.path.join(PROC_DIR, "topic_assignments.csv")
    df[["paper_id", "title", "year", "topic_id", "topic_label", "topic_prob"]].to_csv(
        assign_out,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  → 토픽 할당 저장: {assign_out}")

    # 2) 토픽 정보 저장
    topic_out = os.path.join(PROC_DIR, "topic_info.csv")
    topic_info.to_csv(topic_out, index=False, encoding="utf-8-sig")
    print(f"  → 토픽 정보 저장: {topic_out}")

    # 3) 연도별 토픽 변화 저장
    try:
        topics_over_time = topic_model.topics_over_time(
            abstracts,
            timestamps,
            nr_bins=10,
        )

        tot_out = os.path.join(PROC_DIR, "topic_over_time.csv")
        topics_over_time.to_csv(tot_out, index=False, encoding="utf-8-sig")
        print(f"  → 시계열 토픽 저장: {tot_out}")

    except Exception as e:
        print(f"  ⚠ Dynamic Topic Modeling 실패: {e}")

    # 4) 모델 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, "bertopic_model.pkl")
    topic_model.save(model_path)

    print(f"  → 모델 저장: {model_path}")

    return topic_model, df


def print_topics(topic_model):
    print("\n[ 주요 토픽 목록 ]")

    info = topic_model.get_topic_info()

    for _, row in info[info["Topic"] != -1].head(20).iterrows():
        topic_words = topic_model.get_topic(row["Topic"])

        if topic_words is None:
            continue

        words = ", ".join([word for word, _ in topic_words[:7]])
        print(f"  Topic {row['Topic']:>3} ({row['Count']:>4}건): {words}")


def main():
    print("=" * 60)
    print(" BERTopic 토픽 모델링")
    print("=" * 60)

    inp = os.path.join(PROC_DIR, "papers_clean.csv")
    df = pd.read_csv(inp, low_memory=False)

    print(f"  입력: {len(df):,}건")

    required_cols = ["paper_id", "title", "year", "published", "abstract_clean"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    topic_model, df = run_bertopic(df)
    print_topics(topic_model)

    print("\nBERTopic 분석 완료")
    print("다음 단계: python 3_analyze/cooccurrence.py")
    print("그 다음: python 3_analyze/embedding_similarity.py")


if __name__ == "__main__":
    main()