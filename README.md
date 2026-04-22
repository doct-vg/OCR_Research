# OCR 기술 연관성 분석 프로젝트

## 프로젝트 구조
```
ocr_research/
├── README.md
├── requirements.txt
├── config.py                  # 공통 설정
│
├── 1_collect/
│   └── collect_abstracts.py   # 데이터 수집 (arXiv + Semantic Scholar)
│
├── 2_preprocess/
│   └── preprocess.py          # 텍스트 정제 및 필터링
│
├── 3_analyze/
│   ├── cooccurrence.py        # ① 키워드 공출현 분석
│   ├── bertopic_model.py      # ② BERTopic 토픽 모델링
│   ├── embedding_similarity.py # ③ 임베딩 기반 유사도
│   └── citation_network.py    # ④ 인용 네트워크 분석
│
├── 4_visualize/
│   └── visualize.py           # 통합 시각화
│
├── data/
│   ├── raw/                   # 수집된 원본 데이터
│   └── processed/             # 전처리된 데이터
│
└── outputs/                   # 분석 결과 및 그래프
```

## 실행 순서
```bash
pip install -r requirements.txt

python 1_collect/collect_abstracts.py
python 2_preprocess/preprocess.py
python 3_analyze/cooccurrence.py
python 3_analyze/bertopic_model.py
python 3_analyze/embedding_similarity.py
python 3_analyze/citation_network.py
python 4_visualize/visualize.py
```

## 수집 대상
- 기간: 2016년 1월 ~ 2026년 3월
- 키워드: optical character recognition, scene text recognition, text detection, document analysis, handwritten text recognition
- 소스: arXiv API, Semantic Scholar API
