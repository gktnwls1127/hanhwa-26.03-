import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

import reportEncoders
from reportEncoders import NAML


# gpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 장치:", device)

# csv 읽기
df = pd.read_csv("./data/hanhwa_report.csv")

print("행 개수:", len(df))
print(df.head())

titles = df["제목"].astype(str).unique().tolist()
title2id = {t: i for i, t in enumerate(titles)}
print("제목 종류 수:", len(title2id))

# tokenizer word2id실행
def tokenize(text: str):
    if not isinstance(text, str):
        text = str(text)
    return text.split()

word2id = {"<PAD>": 0, "<UNK>": 1}  # 0: 패딩, 1: 모르는 단어

for col in ["시간", "본문"]:
    for text in df[col].astype(str):
        for tok in tokenize(text):
            if tok not in word2id:
                word2id[tok] = len(word2id)

vocab_size = len(word2id)
print("vocab_size:", vocab_size)


def encode(text: str, max_len: int):
    tokens = tokenize(text)
    ids = [word2id.get(t, 1) for t in tokens][:max_len]  # 1 = <UNK>
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))               # 0 = <PAD>
    return ids

max_time_len = 20
max_content_len = 200

N = len(df)          # 보고서 개수
batch_size = 1       # 한 번에 1 배치만 사용
report_num = N       # 한 배치 안에 N개의 보고서를 모두 넣기


title_ids = torch.zeros(batch_size, report_num, dtype=torch.long)
time_text = torch.zeros(batch_size, report_num, max_time_len, dtype=torch.long)
content_text = torch.zeros(batch_size, report_num, max_content_len, dtype=torch.long)
time_mask = torch.ones(batch_size, report_num, max_time_len)
content_mask = torch.ones(batch_size, report_num, max_content_len)


title_ids = title_ids.to(device)
time_text = time_text.to(device)
content_text = content_text.to(device)
time_mask = time_mask.to(device)
content_mask = content_mask.to(device)


for i, row in df.iterrows():
    title_ids[0, i] = title2id[str(row["제목"])]
    time_ids = encode(row["시간"], max_time_len)
    cont_ids = encode(row["본문"], max_content_len)
    time_text[0, i] = torch.tensor(time_ids, dtype=torch.long)
    content_text[0, i] = torch.tensor(cont_ids, dtype=torch.long)

print("입력 텐서 준비 완료.")
print("time_text shape :", time_text.shape)
print("content_text shape :", content_text.shape)


'''
def load_glove(embedding_file, word2id, embedding_dim):

    vocab_size = len(word2id)

    # 기본값: 0으로 채워진 행렬
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")

    found = 0   # GloVe에서 실제로 찾은 단어 수
    total = 0   # GloVe 파일에서 읽은 단어 수

    print(f"GloVe 임베딩 파일 읽는 중: {embedding_file}")

    with open(embedding_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < embedding_dim + 1:
                continue

            word = parts[0]
            vec_str = parts[1:1+embedding_dim]
            total += 1

            if word in word2id:
                idx = word2id[word]
                vec = [float(x) for x in vec_str]
                embedding_matrix[idx] = vec
                found += 1

    # print(f"GloVe에서 찾은 단어 수: {found} / {vocab_size}")
    # print(f"GloVe 파일 안 단어 수(대략): {total}")

    return torch.tensor(embedding_matrix, dtype=torch.float32)


GLOVE_FILE = "glove/glove.6B.300d.txt"
GLOVE_DIM = 300

pretrained_embedding_matrix = load_glove(
    GLOVE_FILE,
    word2id,
    GLOVE_DIM
)
pretrained_embedding_matrix = pretrained_embedding_matrix.to(device)

def simple_reportencoder_init(self, config):
    nn.Module.__init__(self)

    # 1) 단어 임베딩: 우리가 GloVe에서 만든 embedding_matrix를 그대로 사용
    self.word_embedding_dim = pretrained_embedding_matrix.size(1)  # 300
    self.word_embedding = nn.Embedding(
        num_embeddings=pretrained_embedding_matrix.size(0),
        embedding_dim=self.word_embedding_dim
    )
    with torch.no_grad():
        self.word_embedding.weight.copy_(pretrained_embedding_matrix)

    # 2) 제목 임베딩: 제목 종류 수만큼 ID가 있으니 그 크기로
    self.title_embedding = nn.Embedding(
        num_embeddings=len(title2id),
        embedding_dim=50
    )

    # 3) dropout 설정 (그냥 고정 값)
    self.dropout = nn.Dropout(p=0.2, inplace=True)
    self.dropout_ = nn.Dropout(p=0.2, inplace=False)
    self.auxiliary_loss = None


reportEncoders.ReportEncoder.__init__ = simple_reportencoder_init
'''


print("EmbeddingGemma 모델 로드 중...")

MODEL_NAME = "google/embeddinggemma-300M" 

st_model = SentenceTransformer(MODEL_NAME, device=str(device))

print(f"SentenceTransformer Device: {st_model.device}")
print("모델 로드 완료")

def build_embedding_matrix_with_gemma_ST(word2id, model, target_dim=300):
    vocab_size = len(word2id)
    embedding_matrix = np.zeros((vocab_size, target_dim), dtype="float32")

    print("임베딩 생성 시작 (단어 수:", vocab_size, ")")
    for word, idx in word2id.items():
        if word == "<PAD>":
            # 패딩은 0벡터 유지
            continue

        # 한 단어를 짧은 텍스트로 보고 임베딩
        emb = model.encode(word)        # [768] (기본 크기)
        emb = emb[:target_dim]          # 앞 300차원만 사용

        # 길이 1로 정규화 (선택적이지만 보통 안정적)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        embedding_matrix[idx] = emb

        if idx % 500 == 0:
            print(f"  진행 {idx}/{vocab_size}")

    print("Gemma 임베딩 생성 완료.")
    emb_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
    print("embedding mean:", emb_tensor.mean().item())
    print("embedding std :", emb_tensor.std().item())
    print("min/max:", emb_tensor.min().item(), emb_tensor.max().item())
    return emb_tensor


pretrained_embedding_matrix = build_embedding_matrix_with_gemma_ST(
    word2id,
    st_model,
    target_dim=300
)

def gemma_reportencoder_init(self, config):
    # nn.Module 기본 초기화
    nn.Module.__init__(self)

    # 1) 단어 임베딩: Gemma 기반 행렬 사용
    self.word_embedding_dim = pretrained_embedding_matrix.size(1)  # 300
    self.word_embedding = nn.Embedding(
        num_embeddings=pretrained_embedding_matrix.size(0),
        embedding_dim=self.word_embedding_dim
    )
    with torch.no_grad():
        self.word_embedding.weight.copy_(pretrained_embedding_matrix)

    # 2) 제목 임베딩: 제목 종류 수 기준
    self.title_embedding = nn.Embedding(
        num_embeddings=len(title2id),
        embedding_dim=50
    )

    # 3) 드롭아웃 설정
    self.dropout = nn.Dropout(p=0.2, inplace=True)
    self.dropout_ = nn.Dropout(p=0.2, inplace=False)
    self.auxiliary_loss = None

reportEncoders.ReportEncoder.__init__ = gemma_reportencoder_init

class DummyConfig:
    pass

dummy_config = DummyConfig()
dummy_config.vocabulary_size = vocab_size 


encoder = NAML(dummy_config).to(device)
encoder.initialize()  

with torch.no_grad():
    output = encoder(
        title=title_ids,
        content_text=content_text,
        content_mask=content_mask,
        time_text=time_text,
        time_mask=time_mask,
        user_embedding=None  
    )

print("출력 텐서 크기:", output.shape)  #  [8, 4200, 64]
