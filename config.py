"""
config.py - 프로젝트 공통 설정
"""
import os

# ── 경로 설정 ──────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
PROC_DIR    = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

for d in [RAW_DIR, PROC_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 수집 설정 ──────────────────────────────────────────
keywords = [
    'OCR',
    'optical character recognition',
    'scene text recognition',
    'text detection',
    'document analysis',
    'handwritten text recognition'
]

SEARCH_QUERY = " OR ".join(keywords)
START_DATE      = "2016-01-01"
END_DATE        = "2026-03-31"
MAX_ARXIV       = 5000   # arXiv 전체 상한 (안전망)

# ── 월별 수집 상한 ─────────────────────────────────────
# None 으로 설정하면 월별 제한 없이 전체 상한(MAX_ARXIV)만 적용
# 숫자로 설정하면 연-월 단위로 균등 제한
# 예) 50 → 매월 최대 50건  (10년 × 12개월 × 50 = 최대 6,000건)
MAX_PER_MONTH_ARXIV = 50   # arXiv 월별 상한

# ── 분석 설정 ──────────────────────────────────────────
# 1. 공출현


# 2. BERTopic


# 3. 임베딩


# 4. 인용 네트워크


# ── 기술 키워드 사전 (공출현 분석용) ──────────────────
TECH_KEYWORDS = {
    "Deep Learning"         : ["deep learning", "neural network", "cnn", "convolutional"],
    "Transformer / Attention": ["transformer", "attention mechanism", "bert", "vit", "vision transformer"],
    "LSTM / RNN"            : ["lstm", "rnn", "recurrent", "gru", "sequence"],
    "Computer Vision"       : ["computer vision", "image processing", "object detection", "segmentation"],
    "NLP"                   : ["natural language processing", "nlp", "text recognition", "language model"],
    "Scene Text"            : ["scene text", "text detection", "text spotting", "wild"],
    "Handwriting"           : ["handwriting", "handwritten", "cursive", "manuscript"],
    "Document Analysis"     : ["document analysis", "document understanding", "layout", "table detection"],
    "Multilingual"          : ["multilingual", "arabic", "chinese", "hindi", "indic", "script"],
    "Medical / Healthcare"  : ["medical", "clinical", "prescription", "healthcare", "ehr"],
    "Historical Documents"  : ["historical", "degraded", "ancient", "archival", "digitization"],
    "Data Augmentation"     : ["data augmentation", "synthetic", "generation", "gan"],
    "Edge / Mobile"         : ["edge computing", "mobile", "lightweight", "embedded", "real-time"],
    "Federated Learning"    : ["federated learning", "privacy", "distributed"],
    "Large Language Model"  : ["llm", "large language model", "gpt", "chatgpt", "multimodal"],
}