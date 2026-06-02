"""
3_analyze/embedding_similarity.py
───────────────────────────────────
③ 임베딩 기반 의미 유사도 분석
   - SPECTER 모델로 초록 임베딩
   - FAISS로 최근접 이웃 탐색
   - OCR 논문과 타 기술 분야 간의 의미적 거리 측정
   - UMAP 2D 시각화용 좌표 생성

실행: python 3_analyze/embedding_similarity.py
입력: data/processed/papers_clean.csv
      data/processed/papers_tagged.csv
출력: data/processed/embeddings.npy
      data/processed/umap_coords.csv
      data/processed/similarity_scores.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from umap import UMAP
from tqdm import tqdm
from config import PROC_DIR, EMBED_MODEL, EMBED_BATCH, TECH_KEYWORDS


def embed_abstracts(abstracts: list[str], model_name: str) -> np.ndarray:
    """초록을 임베딩 벡터로 변환합니다."""
    print(f"  임베딩 모델 로드: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"  임베딩 생성 중 (배치 크기={EMBED_BATCH})...")
    embeddings = model.encode(
        abstracts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,  # 코사인 유사도를 위해 정규화
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """FAISS 내적(inner product) 인덱스를 생성합니다. (정규화된 벡터 → 코사인 유사도)"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def calc_tech_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
) -> pd.DataFrame:
    """
    각 기술 카테고리의 평균 임베딩을 구하고,
    OCR 논문 임베딩과의 평균 코사인 유사도를 계산합니다.
    """
    # OCR 핵심 논문 식별 (제목에 OCR/optical character 포함)
    ocr_mask = df["title"].str.lower().str.contains(
        "optical character|ocr", na=False, regex=True
    )
    ocr_indices = df[ocr_mask].index.tolist()

    if not ocr_indices:
        print("  ⚠ OCR 핵심 논문을 찾지 못했습니다. 전체를 OCR 코퍼스로 사용합니다.")
        ocr_indices = df.index.tolist()

    ocr_embeddings = embeddings[ocr_indices]
    ocr_centroid   = ocr_embeddings.mean(axis=0, keepdims=True).astype("float32")
    # 정규화
    faiss.normalize_L2(ocr_centroid)

    records = []
    for tech, keywords in TECH_KEYWORDS.items():
        tech_mask = df["abstract_clean"].str.lower().apply(
            lambda t: any(kw in str(t) for kw in keywords)
        )
        tech_indices = df[tech_mask].index.tolist()
        if len(tech_indices) < 3:
            continue

        tech_embs = embeddings[tech_indices].astype("float32")
        tech_centroid = tech_embs.mean(axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(tech_centroid)

        # 내적 = 코사인 유사도 (정규화된 벡터)
        sim = float(np.dot(ocr_centroid, tech_centroid.T)[0, 0])

        records.append({
            "technology"  : tech,
            "paper_count" : len(tech_indices),
            "cos_similarity_to_ocr": round(sim, 4),
        })

    result = pd.DataFrame(records).sort_values("cos_similarity_to_ocr", ascending=False)
    return result


def generate_umap_coords(embeddings: np.ndarray) -> np.ndarray:
    """시각화용 2D UMAP 좌표를 생성합니다."""
    print("  UMAP 2D 좌표 생성 중...")
    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def main():
    print("=" * 60)
    print("  임베딩 기반 의미 유사도 분석")
    print("=" * 60)

    # 데이터 로드
    clean_path  = os.path.join(PROC_DIR, "papers_clean.csv")
    tagged_path = os.path.join(PROC_DIR, "papers_tagged.csv")

    df = pd.read_csv(clean_path, low_memory=False)
    if os.path.exists(tagged_path):
        tagged = pd.read_csv(tagged_path, usecols=["paper_id", "tech_tags"], low_memory=False)
        df = df.merge(tagged, on="paper_id", how="left")

    print(f"  입력: {len(df):,}건")
    abstracts = df["abstract_clean"].fillna("").tolist()

    # ── 임베딩 생성 (캐시 활용) ───────────────────────
    emb_path = os.path.join(PROC_DIR, "embeddings.npy")
    if os.path.exists(emb_path):
        print(f"  임베딩 캐시 로드: {emb_path}")
        embeddings = np.load(emb_path).astype("float32")
    else:
        embeddings = embed_abstracts(abstracts, EMBED_MODEL)
        np.save(emb_path, embeddings)
        print(f"  → 임베딩 저장: {emb_path}")

    # ── FAISS 인덱스 ──────────────────────────────────
    index = build_faiss_index(embeddings)

    # ── 기술 유사도 계산 ──────────────────────────────
    print("\n  OCR ↔ 기술 유사도 계산...")
    sim_df = calc_tech_similarity(df, embeddings, index)
    sim_out = os.path.join(PROC_DIR, "similarity_scores.csv")
    sim_df.to_csv(sim_out, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {sim_out}")

    print("\n[ OCR과 기술 간 의미적 유사도 ]")
    for _, row in sim_df.iterrows():
        bar = "█" * int(row["cos_similarity_to_ocr"] * 30)
        print(f"  {row['technology']:<30} {row['cos_similarity_to_ocr']:.4f}  {bar}")

    # ── UMAP 2D 좌표 ──────────────────────────────────
    coords_path = os.path.join(PROC_DIR, "umap_coords.npy")
    if os.path.exists(coords_path):
        print(f"\n  UMAP 좌표 캐시 로드")
        coords = np.load(coords_path)
    else:
        coords = generate_umap_coords(embeddings)
        np.save(coords_path, coords)

    umap_df = df[["paper_id", "title", "year"]].copy()
    umap_df["umap_x"] = coords[:, 0]
    umap_df["umap_y"] = coords[:, 1]
    if "tech_tags" in df.columns:
        umap_df["tech_tags"] = df["tech_tags"]

    umap_out = os.path.join(PROC_DIR, "umap_coords.csv")
    umap_df.to_csv(umap_out, index=False, encoding="utf-8-sig")
    print(f"  → UMAP 좌표 저장: {umap_out}")

    print("\n 임베딩 유사도 분석 완료")


if __name__ == "__main__":
    main()
