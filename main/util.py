import os
import torch
import torch.nn as nn
import numpy as np
import json
from Command_corpus import Command_Corpus
from Command_dataset import Command_DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring
from torch import Tensor

'''
LIME - Freshness-guided interest refinement

'''
class RemainingLifetimeWeighting(nn.Module):
    def __init__(self, config):
        super(RemainingLifetimeWeighting, self).__init__()
        self.alpha = config.sigmoid_scaling_alpha 
        self.beta = config.penalty_scaling_beta
        self.use_expired_penalty = config.use_expired_penalty
        self.use_remaining_lifetime_weighting = config.use_remaining_lifetime_weighting

    def forward(self, user_embedding, news_embedding, remaining_lifetime):
        """
        user_embedding:    [batch_size, news_num, embedding_dim]
        news_embedding:    [batch_size, news_num, embedding_dim]
        remaining_lifetime: [batch_size, news_num]

        Output: weighted matching score
        """
        # base matching score (dot product)
        base_score = (user_embedding * news_embedding).sum(dim=-1)                  # [batch_size, news_num]

        if not self.use_remaining_lifetime_weighting:
            return base_score


        # Remaining Lifetime-guided Weighting (based on existing LANCER[1] method)
        if self.use_expired_penalty:
            positive_mask = (remaining_lifetime >= 0).float()
            negative_mask = (remaining_lifetime < 0).float()
            weight = torch.sigmoid(self.alpha * remaining_lifetime)
            weight = positive_mask * weight + negative_mask * self.beta * weight
        else:
            # [1] LANCER : A Lifetime-Aware News Recommender System, in AAAI’23
            weight = torch.sigmoid(self.alpha * remaining_lifetime.abs())           # [batch_size, news_num]

        adjusted_score = base_score * weight
        return adjusted_score

    def initialize(self):
        pass


def pairwise_cosine_similarity(x: Tensor, y: Tensor, zero_diagonal: bool = False) -> Tensor:
    r"""
    Calculates the pairwise cosine similarity matrix

    Args:
        x: tensor of shape ``(batch_size, M, d)``
        y: tensor of shape ``(batch_size, N, d)``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    Returns:
        A tensor of shape ``(batch_size, M, N)``
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    distance = torch.matmul(torch.div(x, x_norm), torch.div(y, y_norm).permute(0, 2, 1))
    if zero_diagonal:
        assert x.shape[1] == y.shape[1]
        mask = torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1).bool().to(distance.device)
        distance.masked_fill_(mask, 0)

    return distance


def compute_scores(model: nn.Module, command_corpus: Command_Corpus, config, batch_size: int, mode: str, result_file: str):
    assert mode in ['dev', 'test'], "mode must be chosen from 'dev' or 'test'"

    dataset = config.dataset
    #truth_path = f"{mode}/ref/truth-{dataset}.txt"

    # 시간별 test 평가용 truth 파일 경로
    truth_suffix = '-time' if getattr(config, 'time_eval', False) else ''
    truth_path = f"{mode}/ref/truth-{dataset}{truth_suffix}.txt"

    # truth 라인 수(=impression 수) 확보 + 각 impression의 후보 수(labels 길이)도 확보
    with open(truth_path, "r", encoding="utf-8") as tf:
        truth_lines = tf.readlines()
    impression_num = len(truth_lines)
    truth_sizes = []
    for line in truth_lines:
        _impid, labels_json = line.strip("\n").split()
        truth_sizes.append(len(json.loads(labels_json)))

    # DataLoader: shuffle=False 중요 (behaviors.tsv 순서 유지)
    dataloader = DataLoader(
        Command_DevTest_Dataset(command_corpus, config, mode),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    sub_scores = [[] for _ in range(impression_num)]
    cand_users = [None for _ in range(impression_num)]
    seen_impressions = 0

    if config.gpu_available:
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            (user_idx, user_dept, user_pos, user_rank, user_unit,
             cand_title_text, cand_title_mask,
             cand_content_text, cand_content_mask,
             cand_time_text, cand_time_mask,
             cand_hist_category, cand_hist_mask,
             cand_hist_graph, cand_cat_mask, cand_cat_idx,
             cmd_title_text, cmd_title_mask,
             cmd_content_text, cmd_content_mask,
             cmd_time_text, cmd_time_mask,
             cmd_category,
             sample_idx) = batch  # ★ 여기 sample_idx를 사용해서 impression 매핑

            # 허용 형태:
            # - user_idx: [B]  -> [B,1]로만 보정
            # - user_idx: [B,1] or [B,K] -> 그대로 사용
            if user_idx.dim() == 1:
                # 배치에 impression이 1개인데 후보가 텐서로 안 묶인 특이 케이스 방어용
                user_idx  = user_idx.unsqueeze(1)
                user_dept = user_dept.unsqueeze(1)
                user_pos  = user_pos.unsqueeze(1)
                user_rank = user_rank.unsqueeze(1)
                user_unit = user_unit.unsqueeze(1)

                cand_title_text   = cand_title_text.unsqueeze(1)
                cand_title_mask   = cand_title_mask.unsqueeze(1)
                cand_content_text = cand_content_text.unsqueeze(1)
                cand_content_mask = cand_content_mask.unsqueeze(1)
                cand_time_text    = cand_time_text.unsqueeze(1)
                cand_time_mask    = cand_time_mask.unsqueeze(1)
                cand_hist_category = cand_hist_category.unsqueeze(1)
                cand_hist_mask     = cand_hist_mask.unsqueeze(1)
                if cand_hist_graph is not None: cand_hist_graph = cand_hist_graph.unsqueeze(1)
                if cand_cat_mask  is not None:  cand_cat_mask  = cand_cat_mask.unsqueeze(1)
                if cand_cat_idx   is not None:  cand_cat_idx   = cand_cat_idx.unsqueeze(1)
            elif user_idx.dim() == 2:
                # [B,1]도 OK, [B,K]도 OK
                pass
            else:
                raise RuntimeError(f"[compute_scores] unexpected user_idx dim={user_idx.dim()} shape={tuple(user_idx.shape)}")

            if config.gpu_available:
                user_idx = user_idx.cuda(non_blocking=True)
                user_dept = user_dept.cuda(non_blocking=True)
                user_pos = user_pos.cuda(non_blocking=True)
                user_rank = user_rank.cuda(non_blocking=True)
                user_unit = user_unit.cuda(non_blocking=True)

                cand_title_text = cand_title_text.cuda(non_blocking=True)
                cand_title_mask = cand_title_mask.cuda(non_blocking=True)
                cand_content_text = cand_content_text.cuda(non_blocking=True)
                cand_content_mask = cand_content_mask.cuda(non_blocking=True)
                cand_time_text = cand_time_text.cuda(non_blocking=True)
                cand_time_mask = cand_time_mask.cuda(non_blocking=True)

                cand_hist_category = cand_hist_category.cuda(non_blocking=True)
                cand_hist_mask = cand_hist_mask.cuda(non_blocking=True)

                cand_hist_graph = cand_hist_graph.cuda(non_blocking=True) if cand_hist_graph is not None else None
                cand_cat_mask   = cand_cat_mask.cuda(non_blocking=True) if cand_cat_mask is not None else None
                cand_cat_idx    = cand_cat_idx.cuda(non_blocking=True) if cand_cat_idx is not None else None

                cmd_title_text = cmd_title_text.cuda(non_blocking=True)
                cmd_title_mask = cmd_title_mask.cuda(non_blocking=True)
                cmd_content_text = cmd_content_text.cuda(non_blocking=True)
                cmd_content_mask = cmd_content_mask.cuda(non_blocking=True)
                cmd_time_text = cmd_time_text.cuda(non_blocking=True)
                cmd_time_mask = cmd_time_mask.cuda(non_blocking=True)
                cmd_category = cmd_category.cuda(non_blocking=True)

                sample_idx = sample_idx.cuda(non_blocking=True) if torch.is_tensor(sample_idx) else sample_idx

            # forward -> [B,K] (impression-level) 또는 [B,1] (구형 pair) 모두 허용
            batch_scores = model(
                cmd_title_text, cmd_title_mask,
                cmd_content_text, cmd_content_mask,
                cmd_time_text, cmd_time_mask,
                cmd_category,
                user_idx, user_dept, user_pos, user_rank, user_unit,
                cand_title_text, cand_title_mask,
                cand_content_text, cand_content_mask,
                cand_time_text, cand_time_mask,
                cand_hist_category, cand_hist_mask,
                cand_hist_graph, cand_cat_mask, cand_cat_idx
            )

            if torch.is_tensor(sample_idx):
                imp_ids = sample_idx.detach().cpu().numpy().reshape(-1)
            else:
                imp_ids = np.asarray(sample_idx).reshape(-1)

            scores_np = batch_scores.detach().cpu().numpy()
            if scores_np.ndim == 1:
                # [B] -> [B,1]
                scores_np = scores_np.reshape(-1, 1)
            elif scores_np.ndim == 2:
                pass
            else:
                raise RuntimeError(f"[compute_scores] unexpected batch_scores shape: {scores_np.shape}")

            user_idx_np = user_idx.detach().cpu().numpy()
            if user_idx_np.ndim == 1:
                user_idx_np = user_idx_np.reshape(-1, 1)

            B = scores_np.shape[0]
            if imp_ids.shape[0] != B:
                raise RuntimeError(f"[compute_scores] imp_ids len mismatch: {imp_ids.shape} vs scores {scores_np.shape}")

            for i in range(B):
                imp_idx = int(imp_ids[i])
                if not (0 <= imp_idx < impression_num):
                    raise RuntimeError(
                        f"[compute_scores] imp_idx out of range: {imp_idx} (truth lines={impression_num})"
                    )

                row = scores_np[i].reshape(-1)  # K
                cand_users[imp_idx] = user_idx_np[i].reshape(-1).tolist()  # <-- 핵심: None 방지

                # 후보 순서(원래 포지션) 보존: [score, original_position]
                sub_scores[imp_idx] = [[float(s), j] for j, s in enumerate(row)]

    # ------------------------------------------------------------------
    # 후보 수 검증 (조용히 틀린 지표 방지)
    # ------------------------------------------------------------------
    for i in range(impression_num):
        if len(sub_scores[i]) == 0:
            raise RuntimeError(f"[compute_scores] empty prediction at impression(line) {i+1} (sample_idx 매핑 확인 필요)")
        if len(sub_scores[i]) != truth_sizes[i]:
            raise RuntimeError(
                f"[compute_scores] candidate count mismatch at impression(line) {i+1}: "
                f"pred={len(sub_scores[i])} vs truth={truth_sizes[i]}"
            )

    # ------------------------------------------------------------------
    # result 파일 작성: impid는 무조건 1..N (truth와 일치)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0] * len(sub_score)
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(("" if i == 0 else "\n") + f"{i+1} " + str(result).replace(" ", ""))


    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    with open(truth_path, "r", encoding="utf-8") as truth_f, open(result_file, "r", encoding="utf-8") as result_f:
        auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)

    return auc, mrr, ndcg5, ndcg10


    
def get_run_index(result_dir: str):
    assert os.path.exists(result_dir), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir(result_dir):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open(result_dir + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1

class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)
