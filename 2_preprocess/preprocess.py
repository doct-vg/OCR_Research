"""
2_preprocess/preprocess.py
───────────────────────────
수집된 논문 초록을 정제하고 분석에 맞게 전처리합니다.

실행: python 2_preprocess/preprocess.py
입력: data/raw/merged_papers.csv
출력: data/processed/papers_clean.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from config import RAW_DIR, PROC_DIR

# ── NLTK 리소스 다운로드 (경로 구분 수정) ─────────────────
_NLTK_RESOURCES = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet":   "wordnet",
    "tokenizers/punkt":  "punkt",          # ← punkt는 tokenizers/ 경로
}
for find_path, pkg in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(find_path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ── 정규식 패턴 ────────────────────────────────────────────
# 수식: $...$ 또는 \command{...} 형태만 제거 (일반 중괄호는 건드리지 않음)
RE_MATH    = re.compile(r"\$[^$]*?\$|\\[a-zA-Z]+\{[^}]*?\}")
RE_SPECIAL = re.compile(r"[^a-zA-Z0-9\s\-]")
RE_SPACES  = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────
# 정제 함수
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """초록 텍스트를 정제합니다. (임베딩·BERTopic용)"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = RE_MATH.sub(" ", text)
    text = RE_SPECIAL.sub(" ", text)
    text = RE_SPACES.sub(" ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """토크나이징 + 표제어 추출 + 불용어 제거 (공출현 분석용)"""
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in text.split()
        if len(t) > 2 and t not in STOP_WORDS
    ]
    return tokens


# ─────────────────────────────────────────────────────────
# 전처리 파이프라인
# ─────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    total_in = len(df)
    log = []   # 단계별 제거 건수 기록

    # ── Step 1: 제목 중복 제거 ────────────────────────────
    df["_title_norm"] = df["title"].str.lower().str.strip()
    before = len(df)
    df = df.drop_duplicates(subset="_title_norm", keep="first")
    df = df.drop(columns=["_title_norm"])
    log.append(("제목 중복 제거", before - len(df)))

    # ── Step 2: 연도 범위 필터 ────────────────────────────
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce").fillna(0).astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(1).astype(int)
    before = len(df)
    df = df[(df["year"] >= 2016) & (df["year"] <= 2026)]
    log.append(("연도 범위 외 제거", before - len(df)))

    # ── Step 3: 원본 초록 길이 기준 필터 (200자 미만) ─────
    # clean_text() 적용 전 원본 기준으로 필터링
    # → 수식 제거 후 길이가 짧아지는 케이스도 사전 차단
    before = len(df)
    df = df[df["abstract"].str.len() >= 200]
    log.append(("초록 200자 미만 제거", before - len(df)))

    # ── Step 4: 텍스트 정제 ───────────────────────────────
    tqdm.pandas(desc="  clean_text")
    df = df.copy()   # SettingWithCopyWarning 방지
    df["abstract_raw"]   = df["abstract"].copy()
    df["abstract_clean"] = df["abstract"].progress_apply(clean_text)

    # ── Step 5: 정제 후 길이 재확인 (50자 미만 제거) ──────
    before = len(df)
    df = df[df["abstract_clean"].str.len() >= 50]
    log.append(("정제 후 초록 50자 미만 제거", before - len(df)))

    # ── Step 6: 토크나이징 ────────────────────────────────
    tqdm.pandas(desc="  tokenize ")
    df["tokens"] = df["abstract_clean"].progress_apply(
        lambda t: " ".join(tokenize_and_lemmatize(t))
    )

    # ── Step 7: 분석용 파생 컬럼 ─────────────────────────
    df["abstract_len"] = df["abstract_clean"].str.len()
    df["year_month"]   = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )

    df = df.reset_index(drop=True)

    # ── 정제 요약 출력 ────────────────────────────────────
    print(f"\n  {'단계':<28} {'제거':>6}건")
    print(f"  {'─'*36}")
    for step, n in log:
        flag = " error" if n > 0 else ""
        print(f"  {step:<28} {n:>6}건{flag}")
    print(f"  {'─'*36}")
    print(f"  {'입력':>28} {total_in:>6}건")
    print(f"  {'최종 출력':>28} {len(df):>6}건")

    return df


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" 전처리 시작")
    print("=" * 60)

    inp = os.path.join(RAW_DIR, "arxiv_papers.csv")
    if not os.path.exists(inp):
        print(f"입력 파일 없음: {inp}")
        print("  먼저 1_collect/collect_abstracts.py 를 실행하세요.")
        return

    df = pd.read_csv(inp, low_memory=False)
    print(f"  입력: {len(df):,}건")

    df = preprocess(df)

    out = os.path.join(PROC_DIR, "papers_clean.csv")
    os.makedirs(PROC_DIR, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n전처리 완료 → {out}")
    print(f"\n  연도 분포:")
    print(df["year"].value_counts().sort_index().to_string())
    print("\n다음 단계: python 3_analyze/cooccurrence.py")


if __name__ == "__main__":
    main()