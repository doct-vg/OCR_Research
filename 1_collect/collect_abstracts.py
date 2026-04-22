import arxiv
import pandas as pd
import calendar
from datetime import datetime

# ─────────────────────────────────────
# 설정
# ─────────────────────────────────────
START_YEAR = 2016
END_YEAR   = 2026
MAX_PER_MONTH = 50

KEYWORDS = [
    "optical character recognition",
    "OCR",
    "scene text recognition",
    "text detection",
    "handwritten text recognition"
]

SEARCH_QUERY = " OR ".join(KEYWORDS)

OUTPUT_FILE = "ocr_papers_monthly.csv"


# ─────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────
def get_month_date_range(year, month):
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}{month:02d}010000"
    end   = f"{year}{month:02d}{last_day:02d}2359"
    return start, end


# ─────────────────────────────────────
# 수집 함수
# ─────────────────────────────────────
def collect_arxiv_monthly():
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3
    )

    records = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):

            start, end = get_month_date_range(year, month)

            query = f"({SEARCH_QUERY}) AND submittedDate:[{start} TO {end}]"

            print(f"\n{year}-{month:02d} 수집 중...")

            search = arxiv.Search(
                query=query,
                max_results=MAX_PER_MONTH * 2,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            count = 0

            try:
                for paper in client.results(search):
                    if count >= MAX_PER_MONTH:
                        break

                    pub = paper.published

                    records.append({
                        "year": pub.year,
                        "month": pub.month,
                        "title": paper.title.strip(),
                        "abstract": paper.summary.replace("\n", " "),
                        "authors": ", ".join([a.name for a in paper.authors]),
                        "categories": ", ".join(paper.categories),
                        "url": paper.entry_id
                    })

                    count += 1

                # 한 달치 수집 완료될 때마다 즉시 파일 쓰기(개인 수정)
                if count > 0:
                    temp_df = pd.DataFrame(records)
                    temp_df.to_csv(out, index=False, encoding="utf-8-sig")
                    print(f"  {mk}: {count}건 수집 완료 (누적 {len(records)}건 저장됨)")
                else:
                    print(f"  {mk}: 해당 월 논문 없음")

            except Exception as e:
                print(f"오류 발생: {e}")
                continue

    return pd.DataFrame(records)


# ─────────────────────────────────────
# 실행
# ─────────────────────────────────────
def main():
    df = collect_arxiv_monthly()

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n전체 수집 완료")
    print(f"총 논문 수: {len(df)}")
    print(f"저장 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

