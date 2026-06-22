from Command_unitCorpus import Command_unitCorpus
import time
import numpy as np
from config import Config
import torch.utils.data as data



class Command_Unit_Train_Dataset(data.Dataset):
    def __init__(self, corpus: Command_unitCorpus, config: Config):
        self.corpus = corpus
        self.config = config
        self.negative_sample_num = corpus.negative_sample_num
        self.unit_num = corpus.unit_num

        # command (report)
        self.report_title_text   = corpus.report_title_text
        self.report_title_mask   = corpus.report_title_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_time_text    = corpus.report_time_text
        self.report_time_mask    = corpus.report_time_mask
        self.report_category     = corpus.report_category

        self.unit_size = corpus.unit_size
        self.unit_type = corpus.unit_type
        self.unit_combat_power = corpus.unit_combat_power
        self.unit_location = corpus.unit_location
        self.unit_name = corpus.unit_name

        self.unit_history_index = corpus.train_unit_history_index
        self.unit_history_mask = corpus.train_unit_history_mask

        self.unit_history_graph = corpus.train_unit_history_graph
        self.unit_history_category_mask = corpus.train_unit_history_category_mask
        self.unit_history_category_indices = corpus.train_unit_history_category_indices
        self.unitidx_to_graphrow = corpus.train_unitidx_to_graphrow

        # samples: (cmd_idx, pos_unit, neg_pool, behavior_index)
        self.samples = corpus.train_unitDataset
        self.num = len(self.samples)

        self.train_samples = [
            [0 for _ in range(1 + self.negative_sample_num)]
            for _ in range(self.num)
        ]

    def negative_sampling(self):
        print(f"\nBegin negative sampling, training sample num : {self.num}")
        st = time.time()

        for i, (cmd_idx, pos_unit, neg_pool, behavior_index) in enumerate(self.samples):
            self.train_samples[i][0] = pos_unit

            if neg_pool:
                pool = np.asarray(neg_pool, dtype=np.int64)
                replace = len(pool) < self.negative_sample_num
                negs = np.random.choice(pool, size=self.negative_sample_num, replace=replace)
                self.train_samples[i][1:1 + self.negative_sample_num] = negs

        print(f"End negative sampling, used time : {time.time() - st:.3f}s")

    def __getitem__(self, index):
        cmd_idx, pos_unit, _, behavior_index = self.samples[index]

        units = np.asarray(self.train_samples[index], dtype=np.int64)
        K = units.shape[0]

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
        # 2) candidate units 텐서: K명
        # ----------------------------
        cand_unit_ID = units
        cand_unit_size = self.unit_size[units]
        cand_unit_type = self.unit_type[units]
        cand_combat_power = self.unit_combat_power[units]
        cand_location = self.unit_location[units]
        cand_unit_name = self.unit_name[units]

        cutoff_ts = int(self.corpus.train_behaviors_time_ts.get(behavior_index, 0))
        H = self.config.max_history_num
        hist_idx = np.zeros((K, H), dtype=np.int64)
        hist_mask = np.zeros((K, H), dtype=bool)
        for a, unit_idx in enumerate(units.tolist()):
            hi, hm = self.corpus.get_history_before("train", int(unit_idx), cutoff_ts)
            hist_idx[a] = hi
            hist_mask[a] = hm
        '''
        hist_idx = self.unit_history_index[units].copy()   # [K, H]
        hist_mask = self.unit_history_mask[units].copy()   # [K, H]
        # 현재 커맨드는 history에서 제외
        exclude = (hist_idx == cmd_idx)
        hist_idx[exclude] = 0
        hist_mask[exclude] = False
        '''

        cand_title_text = self.report_title_text[hist_idx]
        cand_title_mask = self.report_title_mask[hist_idx]
        cand_content_text = self.report_content_text[hist_idx]
        cand_content_mask = self.report_content_mask[hist_idx]
        cand_time_text = self.report_time_text[hist_idx]
        cand_time_mask = self.report_time_mask[hist_idx]
        cand_hist_category = self.report_category[hist_idx]

        rows = [self.unitidx_to_graphrow.get(int(unit_idx), 0) for unit_idx in units]
        cand_hist_graph = self.unit_history_graph[rows]
        cand_cat_mask = self.unit_history_category_mask[rows]
        cand_cat_idx = self.unit_history_category_indices[rows]

        pos_index = 0


        return (
            # command
            cmd_title_text, cmd_title_mask,
            cmd_content_text, cmd_content_mask,
            cmd_time_text, cmd_time_mask,
            cmd_category,

            # candidate units
            cand_unit_ID, cand_unit_name, cand_unit_size, cand_unit_type, cand_combat_power, cand_location, 
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
# Dev / Test Dataset (Command → Unit)
# =========================================================

class Command_Unit_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: Command_unitCorpus, config: Config, mode: str):
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

        # ---------- unit meta ----------
        self.unit_size = corpus.unit_size
        self.unit_type = corpus.unit_type
        self.unit_combat_power = corpus.unit_combat_power
        self.unit_location = corpus.unit_location
        self.unit_name = corpus.unit_name

        if mode == "dev":
            self.unit_history_graph = corpus.dev_unit_history_graph
            self.unit_history_category_mask = corpus.dev_unit_history_category_mask
            self.unit_history_category_indices = corpus.dev_unit_history_category_indices
            self.unit_history_index = corpus.dev_unit_history_index
            self.unit_history_mask = corpus.dev_unit_history_mask
            self.samples = corpus.dev_unitDataset
            self.unitidx_to_graphrow = corpus.dev_unitidx_to_graphrow
        else:
            self.unit_history_graph = corpus.test_unit_history_graph
            self.unit_history_category_mask = corpus.test_unit_history_category_mask
            self.unit_history_category_indices = corpus.test_unit_history_category_indices
            self.unit_history_index = corpus.test_unit_history_index
            self.unit_history_mask = corpus.test_unit_history_mask
            self.samples = corpus.test_unitDataset
            self.unitidx_to_graphrow = corpus.test_unitidx_to_graphrow

        self.num = len(self.samples)

    def __getitem__(self, index):
        cmd_idx, unit_idx, behavior_index = self.samples[index]

        if isinstance(unit_idx, (list, tuple)):
            candidate_unit_ID = np.asarray(unit_idx, dtype=np.int64)
        elif hasattr(unit_idx, "dtype") and hasattr(unit_idx, "shape"):
            candidate_unit_ID = np.asarray(unit_idx, dtype=np.int64)
        else:
            candidate_unit_ID = np.asarray([unit_idx], dtype=np.int64)

        rows = [self.unitidx_to_graphrow.get(int(unit_id), 0) for unit_id in candidate_unit_ID ]

        cand_unit_size = self.unit_size[candidate_unit_ID ]
        cand_unit_type = self.unit_type[candidate_unit_ID ]
        cand_combat_power = self.unit_combat_power[candidate_unit_ID ]
        cand_location = self.unit_location[candidate_unit_ID ]
        cand_unit_name = self.unit_name[candidate_unit_ID ]

        if self.mode == "dev":
            cutoff_ts = int(self.corpus.dev_behaviors_time_ts.get(behavior_index, 0))
            mode_str = "dev"
        else:
            cutoff_ts = int(self.corpus.test_behaviors_time_ts.get(behavior_index, 0))
            mode_str = "test"
        H = self.config.max_history_num
        K = candidate_unit_ID .shape[0]
        hist_idx = np.zeros((K, H), dtype=np.int64)
        hist_mask = np.zeros((K, H), dtype=bool)
        for a, unit_id in enumerate(candidate_unit_ID .tolist()):
            hi, hm = self.corpus.get_history_before(mode_str, int(unit_id), cutoff_ts)
            hist_idx[a] = hi
            hist_mask[a] = hm
        '''
        hist_idx = self.unit_history_index[candidate_unit_ID].copy()   # [K, H]
        hist_mask = self.unit_history_mask[candidate_unit_ID].copy()   # [K, H]
        # 현재 커맨드는 history에서 제외
        exclude = (hist_idx == cmd_idx)
        hist_idx[exclude] = 0
        hist_mask[exclude] = False
        '''

        cand_title_text = self.report_title_text[hist_idx]
        cand_title_mask = self.report_title_mask[hist_idx]
        cand_content_text = self.report_content_text[hist_idx]
        cand_content_mask = self.report_content_mask[hist_idx]
        cand_time_text = self.report_time_text[hist_idx]
        cand_time_mask = self.report_time_mask[hist_idx]
        cand_hist_category = self.report_category[hist_idx]

        hist_graph = self.unit_history_graph[rows]
        cat_mask = self.unit_history_category_mask[rows]
        cat_idx = self.unit_history_category_indices[rows]

        cmd_title_text = self.report_title_text[cmd_idx]
        cmd_title_mask = self.report_title_mask[cmd_idx]
        cmd_content_text = self.report_content_text[cmd_idx]
        cmd_content_mask = self.report_content_mask[cmd_idx]
        cmd_time_text = self.report_time_text[cmd_idx]
        cmd_time_mask = self.report_time_mask[cmd_idx]
        cmd_category = self.report_category[cmd_idx]

        return (
            candidate_unit_ID, cand_unit_name,
            cand_unit_size, cand_unit_type, cand_combat_power, cand_location, 
            cand_title_text, cand_title_mask,
            cand_content_text, cand_content_mask,
            cand_time_text, cand_time_mask,
            cand_hist_category,
            hist_mask,
            hist_graph,
            cat_mask,
            cat_idx,

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