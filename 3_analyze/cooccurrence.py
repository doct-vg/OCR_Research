"""
3_analyze/cooccurrence.py
──────────────────────────
① 키워드 공출현 분석
   - TECH_KEYWORDS 사전 기반 기술 태깅
   - 기술 간 공출현 매트릭스 생성
   - 연도별 기술 등장 추이 계산

실행: python 3_analyze/cooccurrence.py
입력: data/processed/papers_clean.csv
출력: data/processed/cooccurrence_matrix.csv
      data/processed/tech_trend_yearly.csv
      data/processed/papers_tagged.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
from config import PROC_DIR, TECH_KEYWORDS


def tag_technologies(abstract: str) -> list[str]:
    """초록에서 언급된 기술 카테고리를 태깅합니다."""
    found = []
    text = abstract.lower()
    for tech, keywords in TECH_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.append(tech)
    return found


def build_cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """기술 간 공출현 빈도 매트릭스를 생성합니다."""
    tech_list = list(TECH_KEYWORDS.keys())
    matrix = defaultdict(lambda: defaultdict(int))

    for tags in tqdm(df["tech_tags"], desc="공출현 집계"):
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split("|") if t.strip()]
        for t1, t2 in combinations(sorted(tags), 2):
            matrix[t1][t2] += 1
            matrix[t2][t1] += 1

    mat_df = pd.DataFrame(matrix, index=tech_list, columns=tech_list).fillna(0).astype(int)
    return mat_df


def build_yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 기술 키워드 등장 비율을 계산합니다."""
    records = []
    for year, grp in df.groupby("year"):
        total = len(grp)
        row = {"year": year, "total_papers": total}
        for tech, keywords in TECH_KEYWORDS.items():
            count = grp["abstract_clean"].apply(
                lambda txt: any(kw in str(txt).lower() for kw in keywords)
            ).sum()
            row[tech] = count
            row[f"{tech}_pct"] = round(count / total * 100, 2) if total > 0 else 0
        records.append(row)

    return pd.DataFrame(records).sort_values("year")


def main():
    print("=" * 60)
    print("  키워드 공출현 분석")
    print("=" * 60)

    inp = os.path.join(PROC_DIR, "papers_clean.csv")
    df  = pd.read_csv(inp, low_memory=False)
    print(f"  입력: {len(df):,}건")

    # 기술 태깅
    tqdm.pandas(desc="기술 태깅")
    df["tech_tags"] = df["abstract_clean"].progress_apply(
        lambda t: "|".join(tag_technologies(str(t)))
    )

    # 태그된 논문 저장
    tagged_out = os.path.join(PROC_DIR, "papers_tagged.csv")
    df.to_csv(tagged_out, index=False, encoding="utf-8-sig")

    # 공출현 매트릭스
    print("\n공출현 매트릭스 생성...")
    mat = build_cooccurrence_matrix(df)
    mat_out = os.path.join(PROC_DIR, "cooccurrence_matrix.csv")
    mat.to_csv(mat_out, encoding="utf-8-sig")
    print(f"  → 저장: {mat_out}")

    # 연도별 트렌드
    print("\n연도별 트렌드 집계...")
    trend = build_yearly_trend(df)
    trend_out = os.path.join(PROC_DIR, "tech_trend_yearly.csv")
    trend.to_csv(trend_out, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {trend_out}")

    # 간단 요약 출력
    print("\n[ 기술별 총 등장 논문 수 ]")
    tag_counts = {}
    for tech in TECH_KEYWORDS:
        n = df["tech_tags"].str.contains(tech, na=False, regex=False).sum()
        tag_counts[tech] = n
    for tech, n in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tech:<30} {n:>5}건")

    print("\n 공출현 분석 완료")
    print("다음 단계: python 3_analyze/bertopic_model.py")


if __name__ == "__main__":
    main()
