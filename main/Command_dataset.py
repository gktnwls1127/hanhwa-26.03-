from Command_corpus import Command_Corpus
import time
import numpy as np
from config import Config
import torch.utils.data as data
from numpy.random import randint


class Command_Train_Dataset(data.Dataset):
    def __init__(self, corpus: Command_Corpus, config: Config):
        self.corpus = corpus
        self.config = config
        self.negative_sample_num = corpus.negative_sample_num
        self.user_num = corpus.user_num

        # command (report)
        self.report_title_text   = corpus.report_title_text
        self.report_title_mask   = corpus.report_title_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_time_text    = corpus.report_time_text
        self.report_time_mask    = corpus.report_time_mask
        self.report_category     = corpus.report_category

        # user meta
        self.user_department = corpus.user_department
        self.user_position   = corpus.user_position
        self.user_rank       = corpus.user_rank
        self.user_unit       = corpus.user_unit

        # user history index/mask (중요!)
        self.user_history_index = corpus.train_user_history_index      # [user_num, max_history]
        self.user_history_mask  = corpus.train_user_history_mask       # [user_num, max_history]

        # graph (optional: ATT는 사실상 안 쓰지만 유지)
        self.user_history_graph            = corpus.train_user_history_graph
        self.user_history_category_mask    = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.useridx_to_graphrow           = corpus.train_useridx_to_graphrow

        # samples: (cmd_idx, pos_user, neg_pool, behavior_index)
        self.samples = corpus.train_userDataset
        self.num = len(self.samples)

        self.train_samples = [
            [0 for _ in range(1 + self.negative_sample_num)]
            for _ in range(self.num)
        ]

    def negative_sampling(self):
        print(f"\nBegin negative sampling, training sample num : {self.num}")
        st = time.time()

        for i, (cmd_idx, pos_user, neg_pool, behavior_index) in enumerate(self.samples):
            self.train_samples[i][0] = pos_user

            if neg_pool:
                pool = np.asarray(neg_pool, dtype=np.int64)

                # 중복 없이 뽑고 싶은데 pool이 작으면 replace=True로 자동 전환
                replace = len(pool) < self.negative_sample_num
                negs = np.random.choice(pool, size=self.negative_sample_num, replace=replace)

                self.train_samples[i][1:1+self.negative_sample_num] = negs


        print(f"End negative sampling, used time : {time.time() - st:.3f}s")

    def __getitem__(self, index):
        cmd_idx, pos_user, _, behavior_index = self.samples[index]

        users = np.asarray(self.train_samples[index], dtype=np.int64)  # [K]
        K = users.shape[0]

        # ----------------------------
        # 1) command(쿼리) 텐서: 1개
        # ----------------------------
        cmd_title_text   = self.report_title_text[cmd_idx]
        cmd_title_mask   = self.report_title_mask[cmd_idx]
        cmd_content_text = self.report_content_text[cmd_idx]
        cmd_content_mask = self.report_content_mask[cmd_idx]
        cmd_time_text    = self.report_time_text[cmd_idx]
        cmd_time_mask    = self.report_time_mask[cmd_idx]
        cmd_category     = self.report_category[cmd_idx]

        # ----------------------------
        # 2) candidate users 텐서: K명
        # ----------------------------
        cand_user_ID   = users
        cand_dept      = self.user_department[users]
        cand_pos       = self.user_position[users]
        cand_rank      = self.user_rank[users]
        cand_unit      = self.user_unit[users]

        # cutoff time (현재 cmd/impression 시간) : behavior_index로 가져옴
        cutoff_ts = int(self.corpus.train_behaviors_time_ts.get(behavior_index, 0))

        H = self.config.max_history_num
        hist_idx  = np.zeros((K, H), dtype=np.int64)
        hist_mask = np.zeros((K, H), dtype=bool)

        # 각 후보 user에 대해 "cutoff 이전 history만" 생성
        for a, uid in enumerate(users.tolist()):
            hi, hm = self.corpus.get_history_before('train', int(uid), cutoff_ts)
            hist_idx[a]  = hi
            hist_mask[a] = hm


        # history의 report 텍스트로 변환: [K, H, ...]
        cand_title_text   = self.report_title_text[hist_idx]
        cand_title_mask   = self.report_title_mask[hist_idx]
        cand_content_text = self.report_content_text[hist_idx]
        cand_content_mask = self.report_content_mask[hist_idx]
        cand_time_text    = self.report_time_text[hist_idx]
        cand_time_mask    = self.report_time_mask[hist_idx]
        cand_hist_category = self.report_category[hist_idx]  # [K, H]

        # graph 관련(있으면 같이)
        rows = [self.useridx_to_graphrow.get(int(uid), 0) for uid in users]
        cand_hist_graph = self.user_history_graph[rows]                 # [K, ...]
        cand_cat_mask   = self.user_history_category_mask[rows]         # [K, ...]
        cand_cat_idx    = self.user_history_category_indices[rows]      # [K, H] (있으면)

        # label: 항상 0번이 pos
        pos_index = 0

        return (
            # command
            cmd_title_text, cmd_title_mask,
            cmd_content_text, cmd_content_mask,
            cmd_time_text, cmd_time_mask,
            cmd_category,

            # candidate users
            cand_user_ID, cand_dept, cand_pos, cand_rank, cand_unit,
            cand_title_text, cand_title_mask,
            cand_content_text, cand_content_mask,
            cand_time_text, cand_time_mask,
            cand_hist_category,
            hist_mask,
            cand_hist_graph,
            cand_cat_mask,
            cand_cat_idx,

            pos_index
        )

    def __len__(self):
        return self.num


# =========================================================
# Dev / Test Dataset (Command → User)
# =========================================================
class Command_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: Command_Corpus, config: Config, mode: str):
        self.corpus = corpus
        self.config = config
        assert mode in ['dev', 'test']
        self.mode = mode

        # ---------- command (report) ----------
        self.report_title_text   = corpus.report_title_text
        self.report_title_mask   = corpus.report_title_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_time_text    = corpus.report_time_text
        self.report_time_mask    = corpus.report_time_mask
        self.report_category     = corpus.report_category

        # ---------- user meta ----------
        self.user_department = corpus.user_department
        self.user_position   = corpus.user_position
        self.user_rank       = corpus.user_rank
        self.user_unit       = corpus.user_unit

        if mode == 'dev':
            self.user_history_graph            = corpus.dev_user_history_graph
            self.user_history_category_mask    = corpus.dev_user_history_category_mask
            self.user_history_category_indices = corpus.dev_user_history_category_indices
            self.user_history_index            = corpus.dev_user_history_index   
            self.user_history_mask             = corpus.dev_user_history_mask    
            self.samples = corpus.dev_userDataset
            self.useridx_to_graphrow = corpus.dev_useridx_to_graphrow
        else:
            self.user_history_graph            = corpus.test_user_history_graph
            self.user_history_category_mask    = corpus.test_user_history_category_mask
            self.user_history_category_indices = corpus.test_user_history_category_indices
            self.user_history_index            = corpus.test_user_history_index  
            self.user_history_mask             = corpus.test_user_history_mask   
            self.samples = corpus.test_userDataset
            self.useridx_to_graphrow = corpus.test_useridx_to_graphrow

        self.num = len(self.samples)

    def __getitem__(self, index):
        # sample = [cmd_idx, user_idx(스칼라 or 리스트/배열), behavior_index]
        cmd_idx, user_idx, behavior_index = self.samples[index]

        # ----------------------------
        # 0) user_idx를 "배열"로 통일 처리
        # ----------------------------
        if isinstance(user_idx, (list, tuple)):
            u = np.asarray(user_idx, dtype=np.int64)
        elif hasattr(user_idx, "dtype") and hasattr(user_idx, "shape"):  # np.ndarray/torch tensor 유사
            u = np.asarray(user_idx, dtype=np.int64)
        else:
            u = np.asarray([user_idx], dtype=np.int64)  # 스칼라도 [1]로

        # graph row: user별로 뽑기
        rows = [self.useridx_to_graphrow.get(int(uid), 0) for uid in u]

        # ----------------------------
        # 1) candidate user meta: [K]
        # ----------------------------
        dept = self.user_department[u]
        pos  = self.user_position[u]
        rank = self.user_rank[u]
        unit = self.user_unit[u]

        # ----------------------------
        # 2) user history index/mask: [K, H]
        # ----------------------------
        # cutoff time (현재 cmd/impression 시간)
        if self.mode == 'dev':
            cutoff_ts = int(self.corpus.dev_behaviors_time_ts.get(behavior_index, 0))
            mode_str = 'dev'
        else:
            cutoff_ts = int(self.corpus.test_behaviors_time_ts.get(behavior_index, 0))
            mode_str = 'test'

        H = self.config.max_history_num
        K = u.shape[0]
        hist_idx  = np.zeros((K, H), dtype=np.int64)
        hist_mask = np.zeros((K, H), dtype=bool)

        for a, uid in enumerate(u.tolist()):
            hi, hm = self.corpus.get_history_before(mode_str, int(uid), cutoff_ts)
            hist_idx[a]  = hi
            hist_mask[a] = hm


        # history -> report text/mask: [K, H, ...]
        user_title_text   = self.report_title_text[hist_idx]
        user_title_mask   = self.report_title_mask[hist_idx]
        user_content_text = self.report_content_text[hist_idx]
        user_content_mask = self.report_content_mask[hist_idx]
        user_time_text    = self.report_time_text[hist_idx]
        user_time_mask    = self.report_time_mask[hist_idx]
        user_hist_category = self.report_category[hist_idx]  # [K, H]

        # ----------------------------
        # 3) graph 관련: [K, ...]
        # ----------------------------
        hist_graph = self.user_history_graph[rows]
        cat_mask   = self.user_history_category_mask[rows]
        cat_idx    = self.user_history_category_indices[rows]

        # ----------------------------
        # 4) command(query) 텐서: 1개
        # ----------------------------
        cmd_title_text   = self.report_title_text[cmd_idx]
        cmd_title_mask   = self.report_title_mask[cmd_idx]
        cmd_content_text = self.report_content_text[cmd_idx]
        cmd_content_mask = self.report_content_mask[cmd_idx]
        cmd_time_text    = self.report_time_text[cmd_idx]
        cmd_time_mask    = self.report_time_mask[cmd_idx]
        cmd_category     = self.report_category[cmd_idx]

        return (
            # user (candidate users)
            u,            # [K]  (기존 user_idx 자리에 들어감)
            dept, pos, rank, unit,
            user_title_text, user_title_mask,
            user_content_text, user_content_mask,
            user_time_text, user_time_mask,
            user_hist_category,
            hist_mask,
            hist_graph,
            cat_mask,
            cat_idx,

            # command
            cmd_title_text,
            cmd_title_mask,
            cmd_content_text,
            cmd_content_mask,
            cmd_time_text,
            cmd_time_mask,
            cmd_category,
            behavior_index
        )

    def __len__(self):
        return self.num