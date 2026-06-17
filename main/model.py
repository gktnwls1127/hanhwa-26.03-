from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import reportEncoders
import userEncoders
from util import RemainingLifetimeWeighting


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        # Report Encoder: 명령 인코딩
        if config.report_encoder == 'LIME':
            if config.content_encoder == 'NAML':
                base_encoder = reportEncoders.NAML(config)
            else:
                raise ValueError(f"Unknown content encoder: {config.content_encoder}")
            self.report_encoder = reportEncoders.LIME(config=config, base_report_encoder=base_encoder)
        elif config.report_encoder == 'NAML':
            self.report_encoder = reportEncoders.NAML(config)
        elif config.report_encoder == 'NAML_noTitle':
            self.report_encoder = reportEncoders.NAML_noTitle(config)
        elif config.report_encoder == 'NAML_noTime':
            self.report_encoder = reportEncoders.NAML_noTime(config)
        elif config.report_encoder == 'NAML_noBody':
            self.report_encoder = reportEncoders.NAML_noBody(config)
        elif config.report_encoder == 'NAML_noCategory':
            self.report_encoder = reportEncoders.NAML_noCategory(config)
        elif config.report_encoder == 'NAML_onlyBody':
            self.report_encoder = reportEncoders.NAML_onlyBody(config)
        elif config.report_encoder == 'MHSA':
            self.report_encoder = reportEncoders.MHSA(config)
        elif config.report_encoder == 'CNN':
            self.report_encoder = reportEncoders.CNN(config)
        elif config.report_encoder == 'CROWN':
            self.report_encoder = reportEncoders.CROWN(config)
        else:
            raise Exception(config.report_encoder + 'is not implemented')

        # User Encoder: 사용자 인코딩 (히스토리 + 속성 정보)
        if config.user_encoder == 'ATT':
            self.user_encoder = userEncoders.ATT(self.report_encoder, config)
        elif config.user_encoder == 'ATT_noPosition':
            self.user_encoder = userEncoders.ATT_noPosition(self.report_encoder, config)
        elif config.user_encoder == 'MHSA':
            self.user_encoder = userEncoders.MHSA(self.report_encoder, config)
        elif config.user_encoder == 'LSTUR':
            self.user_encoder = userEncoders.LSTUR(self.report_encoder, config)
        elif config.user_encoder == 'CROWN':
            self.user_encoder = userEncoders.CROWN(self.report_encoder, config)
        else:
            raise Exception(config.user_encoder + 'is not implemented')
        
        self.report_embedding_dim = self.report_encoder.report_embedding_dim

        self.use_user_embedding = False
        if config.report_encoder == 'LIME':
            self.model_name = config.report_encoder + '-' + config.content_encoder + '-' + config.user_encoder
        else:
            self.model_name = config.report_encoder + '-' + config.user_encoder
        #self.model_name = config.report_encoder + '-' + config.user_encoder
        self.dropout = nn.Dropout(p=config.dropout_rate)

        self.user_embedding = None
        
        self.click_predictor = config.click_predictor
        if self.click_predictor == 'mlp':
            self.mlp = nn.Linear(in_features=self.report_embedding_dim * 2, out_features=self.report_embedding_dim // 2, bias=True)
            self.out = nn.Linear(in_features=self.report_embedding_dim // 2, out_features=1, bias=True)


    def initialize(self):
        self.report_encoder.initialize()
        self.user_encoder.initialize()

        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)

    def forward(self, cmd_title_text, cmd_title_mask, cmd_content_text, cmd_content_mask, cmd_time_text, cmd_time_mask, cmd_category, \
            cand_user_ID, cand_dept, cand_pos, cand_rank, cand_unit, cand_title_text, cand_title_mask, cand_content_text, cand_content_mask, \
                cand_time_text, cand_time_mask, cand_hist_category, cand_hist_mask, cand_hist_graph, cand_cat_mask, cand_cat_idx, _extra=None,):
        B = cand_user_ID.size(0)
        K = cand_user_ID.size(1)

        # 1) command vector: [B, D]
        cmd_title_text   = cmd_title_text.unsqueeze(1)   # [B,1,L]
        cmd_title_mask   = cmd_title_mask.unsqueeze(1)
        cmd_content_text = cmd_content_text.unsqueeze(1)
        cmd_content_mask = cmd_content_mask.unsqueeze(1)
        cmd_time_text    = cmd_time_text.unsqueeze(1)
        cmd_time_mask    = cmd_time_mask.unsqueeze(1)
        cmd_category     = cmd_category.unsqueeze(1)     # [B,1]

        report_representation = self.report_encoder(
            cmd_title_text, cmd_title_mask,
            cmd_content_text, cmd_content_mask,
            cmd_time_text, cmd_time_mask,
            cmd_category,
            None
        ).squeeze(1)  # [B, D]

        # 2) candidate users -> flatten(B*K)로 user_encoder
        BK = B * K

        flat_dept = cand_dept.view(BK)
        flat_pos  = cand_pos.view(BK)
        flat_rank = cand_rank.view(BK)
        flat_unit = cand_unit.view(BK)

        flat_title_text   = cand_title_text.view(BK, *cand_title_text.shape[2:])
        flat_title_mask   = cand_title_mask.view(BK, *cand_title_mask.shape[2:])
        flat_content_text = cand_content_text.view(BK, *cand_content_text.shape[2:])
        flat_content_mask = cand_content_mask.view(BK, *cand_content_mask.shape[2:])
        flat_time_text    = cand_time_text.view(BK, *cand_time_text.shape[2:])
        flat_time_mask    = cand_time_mask.view(BK, *cand_time_mask.shape[2:])
        flat_hist_cat     = cand_hist_category.view(BK, *cand_hist_category.shape[2:])
        flat_hist_mask    = cand_hist_mask.view(BK, *cand_hist_mask.shape[2:])

        flat_hist_graph = cand_hist_graph.view(BK, *cand_hist_graph.shape[2:]) if cand_hist_graph is not None else None
        flat_cat_mask   = cand_cat_mask.view(BK, *cand_cat_mask.shape[2:]) if cand_cat_mask is not None else None
        flat_cat_idx    = cand_cat_idx.view(BK, *cand_cat_idx.shape[2:]) if cand_cat_idx is not None else None

        query_rep = report_representation.unsqueeze(1).repeat(1, K, 1).reshape(B*K, 1, -1)

        user_representation = self.user_encoder(
            flat_dept, flat_pos, flat_rank, flat_unit,
            flat_title_text, flat_title_mask, flat_content_text, flat_content_mask, flat_time_text, flat_time_mask,
            flat_hist_cat, flat_hist_mask, flat_hist_graph, flat_cat_mask, flat_cat_idx, None, query_rep
        ).squeeze(1)  # [BK, D]

        user_representation = user_representation.contiguous().view(B, K, -1)  # [B, K, D]

        # 3) score
        if self.click_predictor == "dot_product":
            logits = (user_representation * report_representation.unsqueeze(1)).sum(dim=2)  # [B, K]
        else:
            # cmd를 [B,K,D]로 확장해서 concat
            cmd_expand = report_representation.unsqueeze(1).expand(-1, K, -1)
            context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, cmd_expand], dim=2)), inplace=True))
            logits = self.out(context).squeeze(2)  # [B, K]

        return logits