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


class UserEncoder(nn.Module):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(UserEncoder, self).__init__()
        
        self.report_embedding_dim = report_encoder.report_embedding_dim
        self.position_embedding = nn.Embedding(num_embeddings=config.position_num, embedding_dim=config.position_embedding_dim)
        self.report_encoder = report_encoder
        self.device = torch.device('cuda')
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)

    # Input (각 배치의 사용자 정보)
    # user_dept                     : [batch_size] 사용자 부서
    # user_pos                      : [batch_size] 사용자 직급 (position)
    # user_rank                     : [batch_size] 사용자 계급
    # user_unit                     : [batch_size] 사용자 부대/팀
    # user_title_text               : [batch_size, max_history_num, max_title_length] 사용자 히스토리 명령들의 제목
    # user_title_mask               : [batch_size, max_history_num, max_title_length]
    # user_content_text             : [batch_size, max_history_num, max_content_length] 사용자 히스토리 명령들의 본문
    # user_content_mask             : [batch_size, max_history_num, max_content_length]
    # user_time_text                : [batch_size, max_history_num, max_time_length] 사용자 히스토리 명령들의 시간
    # user_time_mask                : [batch_size, max_history_num, max_time_length]
    # user_history_category         : [batch_size, max_history_num] 히스토리 명령들의 카테고리
    # user_history_mask             : [batch_size, max_history_num] 실제 히스토리 항목 마스크
    # user_history_graph            : [batch_size, max_history_num, max_history_num] 히스토리 명령 간의 그래프
    # user_history_category_mask    : [batch_size, category_num] 사용자가 읽은 카테고리
    # user_history_category_indices : [batch_size, max_history_num] 히스토리 명령의 카테고리 인덱스
    # user_embedding                : [batch_size, user_embedding_dim] (옵션) 사용자 ID 임베딩
    # candidate_report_representation : [batch_size, candidate_num, report_embedding_dim] 후보 명령들의 임베딩
    # Output
    # user_representation           : [batch_size, candidate_num, report_embedding_dim] 각 후보 명령에 대한 사용자 벡터
    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_report_representation):
        raise Exception('Function forward must be implemented at sub-class')

class MHSA(UserEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(MHSA, self).__init__(report_encoder, config)
        self.multiheadAttention = MultiHeadAttention(
            config.head_num, 
            self.report_embedding_dim, 
            config.max_history_num, 
            config.max_history_num, 
            config.head_dim, 
            config.head_dim
        )
        self.affine = nn.Linear(config.head_num * config.head_dim, self.report_embedding_dim, bias=True)
        self.attention = Attention(self.report_embedding_dim, config.attention_dim)

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()

    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, \
                user_time_text, user_time_mask, user_history_category, user_history_mask, user_history_graph, user_history_category_mask, \
                user_history_category_indices, user_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)  # Number of candidate reports
        # 1. User history report encoding
        history_embedding = None
        history_embedding = self.report_encoder(
            user_title_text, user_title_mask, 
            user_content_text, user_content_mask, 
            user_time_text, user_time_mask, 
            user_history_category, user_embedding
        )  # [batch_size, max_history_num, report_embedding_dim]

        # 2. Multi-Head Self-Attention
        h = self.multiheadAttention(
            history_embedding, history_embedding, history_embedding, user_history_mask
        )  # [batch_size, max_history_num, head_num * head_dim]

        # 3. Linear transformation and activation
        h = F.relu(F.dropout(self.affine(h), training=self.training, inplace=True), inplace=True)  # [batch_size, max_history_num, report_embedding_dim]

        # 4. Attention layer to generate user representation
        user_representation = self.attention(h).unsqueeze(dim=1).repeat(1, report_num, 1)  # [batch_size, report_num, report_embedding_dim]

        return user_representation

class ATT(UserEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(ATT, self).__init__(report_encoder, config)
        self.report_attention = Attention(self.report_embedding_dim, config.attention_dim)
        self.position_affine = nn.Linear(config.position_embedding_dim, self.report_embedding_dim, bias=True)
        self.affine1 = nn.Linear(self.report_embedding_dim, config.attention_dim)
        self.affine2 = nn.Linear(config.attention_dim, 1)


    def initialize(self):
        super().initialize()
        self.report_attention.initialize()
        nn.init.xavier_uniform_(self.position_affine.weight)
        nn.init.zeros_(self.position_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)
        # Step 1) history report encoding
        history_embedding = self.report_encoder(user_title_text, user_title_mask, user_content_text, user_content_mask, 
                                                user_time_text, user_time_mask, user_history_category, user_embedding)  # [batch_size, max_history_num, report_embedding_dim]

        # Step 2) history attention pooling
        history_vector = self.report_attention(history_embedding)  # [batch_size, report_embedding_dim]

        # Step 3) position representation
        position_representation = F.relu(
            self.position_affine(self.position_embedding(user_pos)),
            inplace=True
        )  # [batch_size, report_embedding_dim]

        # Step 4) 후보 report 수만큼 확장
        history_vector = history_vector.unsqueeze(1).expand(-1, report_num, -1)               # [batch_size, report_num, report_embedding_dim]
        position_representation = position_representation.unsqueeze(1).expand(-1, report_num, -1)  # [batch_size, report_num, report_embedding_dim]

        # Step 5) multi-view attention
        feature = torch.stack([history_vector, position_representation], dim=2)  # [batch_size, report_num, 2, report_embedding_dim]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [batch_size, report_num, 2, 1]
        user_representation = (feature * alpha).sum(dim=2, keepdim=False)  # [batch_size, report_num, report_embedding_dim]

        return user_representation

class ATT_noPosition(UserEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(ATT_noPosition, self).__init__(report_encoder, config)
        self.report_attention = Attention(self.report_embedding_dim, config.attention_dim)

    def initialize(self):
        super().initialize()
        self.report_attention.initialize()

    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)
        # Step 1) 히스토리 report 인코딩
        history_embedding = self.report_encoder(user_title_text, user_title_mask, \
                                              user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, user_embedding)            # [batch_size, max_history_num, embedding_dim]
    
        user_representation = self.report_attention(history_embedding).unsqueeze(dim=1).expand(-1, report_num, -1) # [batch_size, report_embedding_dim]
        return user_representation


class LSTUR(UserEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(LSTUR, self).__init__(report_encoder, config)
        self.masking_probability = 1.0 - config.long_term_masking_probability
        self.gru = nn.GRU(self.report_embedding_dim, self.report_embedding_dim, batch_first=True)

    def initialize(self):
        super().initialize()
        for parameter in self.gru.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)

    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask,
        user_time_text, user_time_mask, user_history_category, user_history_mask, user_history_graph, user_history_category_mask, 
        user_history_category_indices, user_embedding, candidate_report_representation):
        batch_size = user_title_text.size(0)
        report_num = candidate_report_representation.size(1)

        if user_embedding is None:
            user_embedding = torch.zeros(
                batch_size, self.report_embedding_dim,
                device=user_title_text.device, dtype=torch.float32
            )

        user_history_num = user_history_mask.sum(dim=1, keepdim=False).long()  # [batch_size]

        # 1) 사용자 히스토리 report 인코딩
        history_embedding = self.report_encoder(
            user_title_text, user_title_mask,
            user_content_text, user_content_mask,
            user_time_text, user_time_mask,
            user_history_category, user_embedding
        )  # [batch_size, max_history_num, report_embedding_dim]

        # 2) 길이 기준 정렬
        sorted_user_history_num, sorted_indices = torch.sort(user_history_num, descending=True)
        _, desorted_indices = torch.sort(sorted_indices, descending=False)
        nonzero_indices = sorted_user_history_num.nonzero(as_tuple=False).squeeze(dim=1)

        # 3) 히스토리 없는 경우
        if nonzero_indices.size(0) == 0:
            return user_embedding.unsqueeze(dim=1).expand(-1, report_num, -1)

        index = nonzero_indices[-1]

        # 4) 히스토리 있는 경우
        if index + 1 == batch_size:
            sorted_user_embedding = user_embedding.index_select(0, sorted_indices)
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(
                    torch.empty([batch_size, 1], device=user_title_text.device).fill_(self.masking_probability)
                )

            sorted_history_embedding = history_embedding.index_select(0, sorted_indices)
            packed_sorted_history_embedding = pack_padded_sequence(
                sorted_history_embedding, sorted_user_history_num.cpu(), batch_first=True
            )
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))
            user_representation = h.squeeze(dim=0).index_select(0, desorted_indices)

        else:
            non_empty_indices = sorted_indices[:index + 1]
            empty_indices = sorted_indices[index + 1:]

            sorted_user_embedding = user_embedding.index_select(0, non_empty_indices)
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(
                    torch.empty([index + 1, 1], device=user_title_text.device).fill_(self.masking_probability)
                )

            sorted_history_embedding = history_embedding.index_select(0, non_empty_indices)
            packed_sorted_history_embedding = pack_padded_sequence(
                sorted_history_embedding, sorted_user_history_num[:index + 1].cpu(), batch_first=True
            )
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))

            user_representation = torch.cat(
                [h.squeeze(dim=0), user_embedding.index_select(0, empty_indices)],
                dim=0
            ).index_select(0, desorted_indices)

        # 5) 후보 report 개수만큼 확장
        user_representation = user_representation.unsqueeze(dim=1).expand(-1, report_num, -1)
        return user_representation

class CROWN(UserEncoder):
    def __init__(self, report_encoder, config):
        super(CROWN, self).__init__(report_encoder, config)
        
        self.attention_dim = config.attention_dim
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))

        # 후보 사용자 1명 + history reports(H개) 그래프 SAGE
        self.graph_sage = GraphSAGE(in_channels = self.report_embedding_dim,
                                    hidden_channels = self.report_embedding_dim,
                                    num_layers = 1,
                                    out_channels = self.report_embedding_dim,
                                    dropout = config.dropout_rate)
        
        # user node embedding
        self.user_node_embedding = nn.Parameter(torch.zeros([1, self.report_embedding_dim]))
        
        # query-aware attention
        self.K = nn.Linear(self.report_embedding_dim, self.attention_dim, bias=False)
        self.Q = nn.Linear(self.report_embedding_dim, self.attention_dim, bias=True)
    
        #self.affine = nn.Linear(self.report_embedding_dim, self.report_embedding_dim, bias=True)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)

        # 사용자 position 정보 결합
        # postition 추가
        self.position_affine = nn.Linear(config.position_embedding_dim, self.report_embedding_dim, bias=True)
        self.affine1 = nn.Linear(self.report_embedding_dim, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
    
    def initialize(self):
        super().initialize()
        nn.init.zeros_(self.user_node_embedding)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)
        # nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.zeros_(self.affine.bias)

        # position 추가
        nn.init.xavier_uniform_(self.position_affine.weight)
        nn.init.zeros_(self.position_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def create_bipartite_graph(self, history_mask, device):
        """
        node 0: user node
        node 1 ~ H: history report nodes
        valid history에 대해서만 user <-> history 양방향 edge 생성
        """
        valid_hist = history_mask.bool()
        hist_indices = torch.arange(1, self.max_history_num + 1, device=device)[valid_hist]

        if hist_indices.numel() == 0:
            
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        else:
            user_nodes = torch.zeros_like(hist_indices)
            src = torch.cat([user_nodes, hist_indices], dim=0)
            dst = torch.cat([hist_indices, user_nodes], dim=0)
            edge_index = torch.stack([src, dst], dim=0)

        return edge_index

    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_report_representation):
        
        """
        - 입력 1개 = 후보 사용자 1명
        - candidate_report_representation = 현재 점수 매길 report(query)
        - 출력 = 현재 report 기준 후보 사용자 representation

        model.py에서 flatten되어 들어오므로:
        batch_size = BK (배치 B x 후보 사용자 K)
        report_num = 1 인 경우가 대부분
        """
        batch_size = user_title_text.size(0)                    # 실제로는 BK (배치 B x 후보 사용자 K)
        report_num = candidate_report_representation.size(1)    # 보통 1

        # --------------------------------------------------
        # 1. 후보 사용자 history reports 인코딩
        # history_embedding: [BK, H, D]
        # --------------------------------------------------
        # 1. history report encoding
        history_embedding = self.report_encoder(
            user_title_text, user_title_mask,
            user_content_text, user_content_mask,
            user_time_text, user_time_mask,
            user_history_category,
            user_embedding
        )  # [BK, H, D]

        # --------------------------------------------------
        # 2. 후보 사용자별 history graph 구성 후 GNN 적용
        # 각 후보 사용자에 대해:
        #   user node 1개 + history nodes H개
        # --------------------------------------------------
        user_rep_list = []

        for i in range(batch_size):
            hist_i = history_embedding[i]              # [H, D]
            user_node = self.dropout_(self.user_node_embedding)  # [1, D]

            # node 0=user, node 1..H=history
            node_feat = torch.cat([user_node, hist_i], dim=0)    # [1+H, D]

            edge_index = self.create_bipartite_graph(
                user_history_mask[i],
                node_feat.device
            )
            # gnn_out: [1+H, D]
            gnn_out = self.graph_sage(node_feat, edge_index)      # [1+H, D]

            # history node만 사용
            hist_out = gnn_out[1:, :]                             # [H, D]

            user_rep_list.append(hist_out)

        gcn_feature = torch.stack(user_rep_list, dim=0)           # [BK, H, D]

        # --------------------------------------------------
        # 3. query-aware attention
        # 현재 report(query)를 기준으로
        # 후보 사용자의 history 중 어떤 것이 중요한지 계산
        # --------------------------------------------------
        gcn_feature = gcn_feature.unsqueeze(1).expand(-1, report_num, -1, -1)  # [BK, R, H, D]

        batch_report_num = batch_size * report_num

        # Key: history, Query: current report
        K = self.K(gcn_feature).view(batch_report_num, self.max_history_num, self.attention_dim)
        Q = self.Q(candidate_report_representation).view(batch_report_num, self.attention_dim, 1)

        # attention score: [BK*R, H]
        a = torch.bmm(K, Q).view(batch_report_num, self.max_history_num) / self.attention_scalar

        hist_mask = user_history_mask.unsqueeze(1).expand(-1, report_num, -1).reshape(batch_report_num, self.max_history_num)
        a = a.masked_fill(~hist_mask.bool(), -1e9)

        alpha = F.softmax(a, dim=1)

        # weighted sum -> history 기반 사용자 표현
        out = torch.bmm(
            alpha.unsqueeze(1),
            gcn_feature.reshape(batch_report_num, self.max_history_num, self.report_embedding_dim)
        )  # [BK*R, 1, D]

        out = out.squeeze(1).view(batch_size, report_num, self.report_embedding_dim)

        # --------------------------------------------------
        # 4. position 정보 기반 표현
        # --------------------------------------------------
        position_representation = F.relu(self.position_affine(self.position_embedding(user_pos)), inplace=True)
        position_representation = position_representation.unsqueeze(1).expand(-1, report_num, -1)

        # --------------------------------------------------
        # 5. multi-view attention
        # history view + position view 결합
        # --------------------------------------------------
        feature = torch.stack([out, position_representation], dim=2)   # [B, R, 2, D]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [B, R, 2, 1]
        user_representation = (feature * alpha).sum(dim=2)   # [B, R, D]

        return user_representation                                                       # [batch_size, report_num, report_embedding_dim]
