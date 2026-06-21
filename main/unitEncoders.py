import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GCN
from torch.nn.utils.rnn import pack_padded_sequence
from layers import MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention, GCN
from reportEncoders import ReportEncoder
from torch_scatter import scatter_sum, scatter_softmax # need to be installed by following `https://pytorch-scatter.readthedocs.io/en/latest`


class UnitEncoder(nn.Module):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(UnitEncoder, self).__init__()
        
        self.report_embedding_dim = report_encoder.report_embedding_dim
        self.unit_name_embedding = nn.Embedding(num_embeddings=config.unit_name_num, embedding_dim=config.unit_name_embedding_dim)
        self.unit_type_embedding  = nn.Embedding(num_embeddings=config.unit_type_num, embedding_dim=config.unit_type_embedding_dim)
        self.report_encoder = report_encoder
        self.device = torch.device('cuda')
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.unit_name_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.unit_type_embedding.weight, -0.1, 0.1)

    # Input (각 배치의 부대 정보)
    # unit_name                     : [batch_size] 부대 이름
    # unit_type                     : [batch_size] 부대 타입 (type)
    # unit_title_text               : [batch_size, max_history_num, max_title_length] 부대 히스토리 명령들의 제목
    # unit_title_mask               : [batch_size, max_history_num, max_title_length]
    # unit_content_text             : [batch_size, max_history_num, max_content_length] 부대 히스토리 명령들의 본문
    # unit_content_mask             : [batch_size, max_history_num, max_content_length]
    # unit_time_text                : [batch_size, max_history_num, max_time_length] 부대 히스토리 명령들의 시간
    # unit_time_mask                : [batch_size, max_history_num, max_time_length]
    # unit_history_category         : [batch_size, max_history_num] 히스토리 명령들의 카테고리
    # unit_history_mask             : [batch_size, max_history_num] 실제 히스토리 항목 마스크
    # unit_history_graph            : [batch_size, max_history_num, max_history_num] 히스토리 명령 간의 그래프
    # unit_history_category_mask    : [batch_size, category_num] 부대가 읽은 카테고리
    # unit_history_category_indices : [batch_size, max_history_num] 히스토리 명령의 카테고리 인덱스
    # unit_embedding                : [batch_size, unit_embedding_dim] (옵션) 부대 ID 임베딩
    # candidate_report_representation : [batch_size, candidate_num, report_embedding_dim] 후보 명령들의 임베딩
    # Output
    # unit_representation           : [batch_size, candidate_num, report_embedding_dim] 각 후보 명령에 대한 부대 벡터
    def forward(self, unit_name, unit_size, unit_type, combat_power, location, unit_title_text, unit_title_mask, unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, \
                unit_history_mask, unit_history_graph, unit_history_category_mask, unit_history_category_indices, unit_embedding, candidate_report_representation):
        raise Exception('Function forward must be implemented at sub-class')

class ATT(UnitEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(ATT, self).__init__(report_encoder, config)
        self.report_attention = Attention(self.report_embedding_dim, config.attention_dim)
        self.unit_name_affine = nn.Linear(config.unit_name_embedding_dim, self.report_embedding_dim, bias=True)
        self.unit_type_affine = nn.Linear(config.unit_type_embedding_dim, self.report_embedding_dim, bias=True)
        self.affine1 = nn.Linear(self.report_embedding_dim, config.attention_dim)
        self.affine2 = nn.Linear(config.attention_dim, 1)


    def initialize(self):
        super().initialize()
        self.report_attention.initialize()
        nn.init.xavier_uniform_(self.unit_name_affine.weight)
        nn.init.zeros_(self.unit_name_affine.bias)
        nn.init.xavier_uniform_(self.unit_type_affine.weight)
        nn.init.zeros_(self.unit_type_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, unit_name, unit_size, unit_type, combat_power, location, unit_title_text, unit_title_mask, unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, \
                unit_history_mask, unit_history_graph, unit_history_category_mask, unit_history_category_indices, unit_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)
        # Step 1) history report encoding
        history_embedding = self.report_encoder(unit_title_text, unit_title_mask, unit_content_text, unit_content_mask, 
                                                unit_time_text, unit_time_mask, unit_history_category, unit_embedding)  # [batch_size, max_history_num, report_embedding_dim]

        # Step 2) history attention pooling
        history_vector = self.report_attention(history_embedding)  # [batch_size, report_embedding_dim]

        # Step 3) unit name representation
        unit_name_representation = F.relu(
            self.unit_name_affine(self.unit_name_embedding(unit_name)),
            inplace=True
        )  # [batch_size, report_embedding_dim]

        unit_type_representation = F.relu(
            self.unit_type_affine(self.unit_type_embedding(unit_type)),
            inplace=True
        )  # [batch_size, report_embedding_dim]

        # Step 4) 후보 report 수만큼 확장
        history_vector = history_vector.unsqueeze(1).expand(-1, report_num, -1)               # [batch_size, report_num, report_embedding_dim]
        unit_name_representation = unit_name_representation.unsqueeze(1).expand(-1, report_num, -1)  # [batch_size, report_num, report_embedding_dim]
        unit_type_representation = unit_type_representation.unsqueeze(1).expand(-1, report_num, -1)  # [batch_size, report_num, report_embedding_dim]

        # Step 5) multi-view attention
        feature = torch.stack([history_vector, unit_name_representation, unit_type_representation], dim=2)  # [batch_size, report_num, 3, report_embedding_dim]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [batch_size, report_num, 3, 1]
        unit_representation = (feature * alpha).sum(dim=2, keepdim=False)  # [batch_size, report_num, report_embedding_dim]

        return unit_representation
    
class ATT_noName(UnitEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(ATT_noName, self).__init__(report_encoder, config)
        self.report_attention = Attention(self.report_embedding_dim, config.attention_dim)
        self.unit_type_affine = nn.Linear(config.unit_type_embedding_dim, self.report_embedding_dim, bias=True)
        self.affine1 = nn.Linear(self.report_embedding_dim, config.attention_dim)
        self.affine2 = nn.Linear(config.attention_dim, 1)

    def initialize(self):
        super().initialize()
        self.report_attention.initialize()
        nn.init.xavier_uniform_(self.unit_type_affine.weight)
        nn.init.zeros_(self.unit_type_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, unit_name, unit_size, unit_type, combat_power, location, unit_title_text, unit_title_mask, unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, \
                unit_history_mask, unit_history_graph, unit_history_category_mask, unit_history_category_indices, unit_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)
        # Step 1) 히스토리 report 인코딩
        history_embedding = self.report_encoder(unit_title_text, unit_title_mask, \
                                              unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, unit_embedding)            # [batch_size, max_history_num, embedding_dim]
    
        # Step 2) history attention pooling
        history_vector = self.report_attention(history_embedding)  # [batch_size, report_embedding_dim]

        # Step 3) unit type representation
        unit_type_representation = F.relu(
            self.unit_type_affine(self.unit_type_embedding(unit_type)),
            inplace=True
        )  # [batch_size, report_embedding_dim]

        # Step 4) 후보 report 수만큼 확장
        history_vector = history_vector.unsqueeze(1).expand(-1, report_num, -1)               # [batch_size, report_num, report_embedding_dim]
        unit_type_representation = unit_type_representation.unsqueeze(1).expand(-1, report_num, -1)  # [batch_size, report_num, report_embedding_dim]

        # Step 5) multi-view attention
        feature = torch.stack([history_vector, unit_type_representation], dim=2)  # [batch_size, report_num, 2, report_embedding_dim]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [batch_size, report_num, 2, 1]
        unit_representation = (feature * alpha).sum(dim=2, keepdim=False)  # [batch_size, report_num, report_embedding_dim]

        return unit_representation

class ATT_noType(UnitEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(ATT_noType, self).__init__(report_encoder, config)
        self.report_attention = Attention(self.report_embedding_dim, config.attention_dim)
        self.unit_name_affine = nn.Linear(config.unit_name_embedding_dim, self.report_embedding_dim, bias=True)
        self.affine1 = nn.Linear(self.report_embedding_dim, config.attention_dim)
        self.affine2 = nn.Linear(config.attention_dim, 1)

    def initialize(self):
        super().initialize()
        self.report_attention.initialize()
        nn.init.xavier_uniform_(self.unit_name_affine.weight)
        nn.init.zeros_(self.unit_name_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, unit_name, unit_size, unit_type, combat_power, location, unit_title_text, unit_title_mask, unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, \
                unit_history_mask, unit_history_graph, unit_history_category_mask, unit_history_category_indices, unit_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)
        # Step 1) 히스토리 report 인코딩
        history_embedding = self.report_encoder(unit_title_text, unit_title_mask, \
                                              unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, unit_embedding)            # [batch_size, max_history_num, embedding_dim]
    
        # Step 2) history attention pooling
        history_vector = self.report_attention(history_embedding)  # [batch_size, report_embedding_dim]

        # Step 3) unit name representation
        unit_name_representation = F.relu(
            self.unit_name_affine(self.unit_name_embedding(unit_name)),
            inplace=True
        )  # [batch_size, report_embedding_dim]

        # Step 4) 후보 report 수만큼 확장
        history_vector = history_vector.unsqueeze(1).expand(-1, report_num, -1)               # [batch_size, report_num, report_embedding_dim]
        unit_name_representation = unit_name_representation.unsqueeze(1).expand(-1, report_num, -1)  # [batch_size, report_num, report_embedding_dim]
        
        # Step 5) multi-view attention
        feature = torch.stack([history_vector, unit_name_representation], dim=2)  # [batch_size, report_num, 3, report_embedding_dim]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [batch_size, report_num, 3, 1]
        unit_representation = (feature * alpha).sum(dim=2, keepdim=False)  # [batch_size, report_num, report_embedding_dim]

        return unit_representation


class ATT_noNameType(UnitEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(ATT_noNameType, self).__init__(report_encoder, config)
        self.report_attention = Attention(self.report_embedding_dim, config.attention_dim)

    def initialize(self):
        super().initialize()
        self.report_attention.initialize()

    def forward(self, unit_name, unit_size, unit_type, combat_power, location, unit_title_text, unit_title_mask, unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, \
                unit_history_mask, unit_history_graph, unit_history_category_mask, unit_history_category_indices, unit_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)
        # Step 1) 히스토리 report 인코딩
        history_embedding = self.report_encoder(unit_title_text, unit_title_mask, \
                                              unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, unit_embedding)            # [batch_size, max_history_num, embedding_dim]
    
        unit_representation = self.report_attention(history_embedding).unsqueeze(dim=1).expand(-1, report_num, -1) # [batch_size, report_embedding_dim]
        return unit_representation

class CROWN(UnitEncoder):
    def __init__(self, report_encoder, config):
        super(CROWN, self).__init__(report_encoder, config)
        
        self.attention_dim = config.attention_dim
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))

        # 후보 부대 1명 + history reports(H개) 그래프 SAGE
        self.graph_sage = GraphSAGE(in_channels = self.report_embedding_dim,
                                    hidden_channels = self.report_embedding_dim,
                                    num_layers = 1,
                                    out_channels = self.report_embedding_dim,
                                    dropout = config.dropout_rate)
        
        # unit node embedding
        self.unit_node_embedding = nn.Parameter(torch.zeros([1, self.report_embedding_dim]))
        
        # query-aware attention
        self.K = nn.Linear(self.report_embedding_dim, self.attention_dim, bias=False)
        self.Q = nn.Linear(self.report_embedding_dim, self.attention_dim, bias=True)
    
        #self.affine = nn.Linear(self.report_embedding_dim, self.report_embedding_dim, bias=True)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)

        # 부대 name, type 정보 결합
        # name, type 추가
        self.unit_name_affine = nn.Linear(config.unit_name_embedding_dim, self.report_embedding_dim, bias=True)
        self.unit_type_affine = nn.Linear(config.unit_type_embedding_dim, self.report_embedding_dim, bias=True)
        self.affine1 = nn.Linear(self.report_embedding_dim, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
    
    def initialize(self):
        super().initialize()
        nn.init.zeros_(self.unit_node_embedding)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)
        # nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.zeros_(self.affine.bias)

        # name, type 추가
        nn.init.xavier_uniform_(self.unit_name_affine.weight)
        nn.init.zeros_(self.unit_name_affine.bias)
        nn.init.xavier_uniform_(self.unit_type_affine.weight)
        nn.init.zeros_(self.unit_type_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def create_bipartite_graph(self, history_mask, device):
        """
        node 0: unit node
        node 1 ~ H: history report nodes
        valid history에 대해서만 unit <-> history 양방향 edge 생성
        """
        valid_hist = history_mask.bool()
        hist_indices = torch.arange(1, self.max_history_num + 1, device=device)[valid_hist]

        if hist_indices.numel() == 0:
            
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        else:
            unit_nodes = torch.zeros_like(hist_indices)
            src = torch.cat([unit_nodes, hist_indices], dim=0)
            dst = torch.cat([hist_indices, unit_nodes], dim=0)
            edge_index = torch.stack([src, dst], dim=0)

        return edge_index

    def forward(self, unit_name, unit_size, unit_type, combat_power, location, unit_title_text, unit_title_mask, unit_content_text, unit_content_mask, unit_time_text, unit_time_mask, unit_history_category, \
                unit_history_mask, unit_history_graph, unit_history_category_mask, unit_history_category_indices, unit_embedding, candidate_report_representation):
        
        """
        - 입력 1개 = 후보 부대 1명
        - candidate_report_representation = 현재 점수 매길 report(query)
        - 출력 = 현재 report 기준 후보 부대 representation

        model.py에서 flatten되어 들어오므로:
        batch_size = BK (배치 B x 후보 부대 K)
        report_num = 1 인 경우가 대부분
        """
        batch_size = unit_title_text.size(0)                    # 실제로는 BK (배치 B x 후보 부대 K)
        report_num = candidate_report_representation.size(1)    # 보통 1

        # --------------------------------------------------
        # 1. 후보 부대 history reports 인코딩
        # history_embedding: [BK, H, D]
        # --------------------------------------------------
        # 1. history report encoding
        history_embedding = self.report_encoder(
            unit_title_text, unit_title_mask,
            unit_content_text, unit_content_mask,
            unit_time_text, unit_time_mask,
            unit_history_category,
            unit_embedding
        )  # [BK, H, D]

        # --------------------------------------------------
        # 2. 후보 부대별 history graph 구성 후 GNN 적용
        # 각 후보 부대에 대해:
        #   unit node 1개 + history nodes H개
        # --------------------------------------------------
        unit_rep_list = []

        for i in range(batch_size):
            hist_i = history_embedding[i]              # [H, D]
            unit_node = self.dropout_(self.unit_node_embedding)  # [1, D]

            # node 0=unit, node 1..H=history
            node_feat = torch.cat([unit_node, hist_i], dim=0)    # [1+H, D]

            edge_index = self.create_bipartite_graph(
                unit_history_mask[i],
                node_feat.device
            )
            # gnn_out: [1+H, D]
            gnn_out = self.graph_sage(node_feat, edge_index)      # [1+H, D]

            # history node만 사용
            hist_out = gnn_out[1:, :]                             # [H, D]

            unit_rep_list.append(hist_out)

        gcn_feature = torch.stack(unit_rep_list, dim=0)           # [BK, H, D]

        # --------------------------------------------------
        # 3. query-aware attention
        # 현재 report(query)를 기준으로
        # 후보 부대의 history 중 어떤 것이 중요한지 계산
        # --------------------------------------------------
        gcn_feature = gcn_feature.unsqueeze(1).expand(-1, report_num, -1, -1)  # [BK, R, H, D]

        batch_report_num = batch_size * report_num

        # Key: history, Query: current report
        K = self.K(gcn_feature).view(batch_report_num, self.max_history_num, self.attention_dim)
        Q = self.Q(candidate_report_representation).view(batch_report_num, self.attention_dim, 1)

        # attention score: [BK*R, H]
        a = torch.bmm(K, Q).view(batch_report_num, self.max_history_num) / self.attention_scalar

        hist_mask = unit_history_mask.unsqueeze(1).expand(-1, report_num, -1).reshape(batch_report_num, self.max_history_num)
        a = a.masked_fill(~hist_mask.bool(), -1e9)

        alpha = F.softmax(a, dim=1)

        # weighted sum -> history 기반 부대 표현
        out = torch.bmm(
            alpha.unsqueeze(1),
            gcn_feature.reshape(batch_report_num, self.max_history_num, self.report_embedding_dim)
        )  # [BK*R, 1, D]

        out = out.squeeze(1).view(batch_size, report_num, self.report_embedding_dim)

        # --------------------------------------------------
        # 4. name, type 정보 기반 표현
        # --------------------------------------------------
        unit_name_representation = F.relu(self.unit_name_affine(self.unit_name_embedding(unit_name)), inplace=True)
        unit_type_representation = F.relu(self.unit_type_affine(self.unit_type_embedding(unit_type)), inplace=True)
        unit_name_representation = unit_name_representation.unsqueeze(1).expand(-1, report_num, -1)
        unit_type_representation = unit_type_representation.unsqueeze(1).expand(-1, report_num, -1)

        # --------------------------------------------------
        # 5. multi-view attention
        # history view + name/type view 결합
        # --------------------------------------------------
        feature = torch.stack([out, unit_name_representation, unit_type_representation], dim=2)   # [B, R, 3, D]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [B, R, 3, 1]
        unit_representation = (feature * alpha).sum(dim=2)   # [B, R, D]

        return unit_representation                                                       # [batch_size, report_num, report_embedding_dim]
