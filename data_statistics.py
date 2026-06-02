import pandas as pd

df = pd.read_csv("data/raw/arxiv_papers.csv", low_memory=False)

# 1. 기본 현황
print(f"총 논문 수: {len(df):,}")
print(f"\n결측값 현황:")
print(df.isnull().sum())

# 2. 초록 길이 분포
df["abstract_len"] = df["abstract"].str.len()
print(f"\n초록 길이 통계:")
print(df["abstract_len"].describe())
print(f"50자 미만: {(df['abstract_len'] < 50).sum()}건")
print(f"200자 미만: {(df['abstract_len'] < 200).sum()}건")

# 3. 연도 분포
print(f"\n연도 분포:")
print(df["year"].value_counts().sort_index())

# 4. 중복 확인
dup_title = df["title"].str.lower().str.strip().duplicated().sum()
dup_id    = df["paper_id"].duplicated().sum()
print(f"\n제목 중복: {dup_title}건")
print(f"paper_id 중복: {dup_id}건")
