import os
import pickle
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention


class ReportEncoder(nn.Module):
    #def __init__(self, config: Config):
    def __init__(self, config):
        super(ReportEncoder, self).__init__()
        self.category_num = config.category_num
        self.word_embedding_dim = config.word_embedding_dim
        # 단어 임베딩 로드 
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        if getattr(config, "time_eval", False):
            cache_dir = os.path.join("cache", "time", config.dataset, "all")
            cache_dataset = f"{config.dataset}-time"
        else:
            cache_dir = os.path.join("cache", "normal", config.dataset)
            cache_dataset = config.dataset
        word_embedding_file = os.path.join(cache_dir, 'word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + cache_dataset  + '.pkl')
        with open(word_embedding_file, 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        # 카테고리 임베딩
        self.category_embedding  = nn.Embedding(num_embeddings=config.category_num + 1, embedding_dim=config.category_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None
        '''
        self.word_embedding_dim = 300
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(3) + '-' + str(300)  + '-' + str(200) + '-' + str(20) + '-' + 'command' + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=20, embedding_dim=50)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.dropout_ = nn.Dropout(p=0.2, inplace=False)
        self.auxiliary_loss = None
        '''

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    # Input
    # title_text          : [batch_size, report_num, max_title_length] 제목
    # title_mask          : [batch_size, report_num, max_title_length]
    # content_text        : [batch_size, report_num, max_content_length] 본문
    # content_mask        : [batch_size, report_num, max_content_length]
    # time_text           : [batch_size, report_num, max_time_length] 시간 정보
    # time_mask           : [batch_size, report_num, max_time_length]
    # category            : [batch_size, report_num] 카테고리
    # user_embedding      : [batch_size, user_embedding_dim] (옵션) 사용자 임베딩
    # Output
    # report_representation : [batch_size, report_num, report_embedding_dim] 명령 임베딩
    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # report_representation : [batch_size, report_num, unfused_report_embedding_dim]
    # category                 : [batch_size, report_num]
    # Output
    # report_representation : [batch_size, report_num, report_embedding_dim]
    def feature_fusion(self, report_representation, category):
        category_representation = self.category_embedding(category)                                                      # [batch_size, report_num, category_embedding_dim]
        report_representation = torch.cat([report_representation, self.dropout(category_representation)], dim=2)   # [batch_size, report_num, report_embedding_dim]
        return report_representation


class MHSA(ReportEncoder):

    def __init__(self, config):
        super(MHSA, self).__init__(config)

        self.max_sentence_length = config.max_title_length
        self.max_content_length = config.max_content_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.report_embedding_dim = 3 * self.feature_dim + config.category_embedding_dim

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)
        report_num = title_text.size(1)
        batch_report_num = batch_size * report_num

        title_mask = title_mask.view([batch_report_num, self.max_sentence_length]) # [batch_size * report_num, max_sentence_length]
        time_mask  = time_mask.view([batch_report_num, self.max_sentence_length])           # [BR,32]
        content_mask = content_mask.view([batch_report_num, self.max_content_length])       # [BR,128]

        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_report_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * report_num, max_sentence_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_report_num, self.max_content_length, self.word_embedding_dim]) # [batch_size * report_num, max_sentence_length, word_embedding_dim]
        time_w = self.dropout(self.word_embedding(time_text)).view([batch_report_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * report_num, max_sentence_length, word_embedding_dim]

        content_w = content_w.view(batch_report_num, 32, 4, self.word_embedding_dim).mean(dim=2)
        content_mask = content_mask.view(batch_report_num, 32, 4).max(dim=2)[0]

        # 2. multi-head self-attention
        title_c = self.dropout(self.multiheadAttention(title_w, title_w, title_w, title_mask))                                                                    # [batch_size * report_num, max_sentence_length, report_embedding_dim]
        content_c = self.dropout(self.multiheadAttention(content_w, content_w, content_w, content_mask))                                                                    # [batch_size * report_num, max_content_length, report_embedding_dim]
        time_c = self.dropout(self.multiheadAttention(time_w, time_w, time_w, time_mask))                                                                    # [batch_size * report_num, max_sentence_length, report_embedding_dim]
        
        # 3. attention layer
        title_representation = self.attention(title_c, mask=title_mask).view([batch_size, report_num, self.feature_dim])                           # [batch_size, report_num, report_embedding_dim]
        content_representation = self.attention(content_c, mask=content_mask).view([batch_size, report_num, self.feature_dim])                           # [batch_size, report_num, report_embedding_dim]
        time_representation = self.attention(time_c, mask=time_mask).view([batch_size, report_num, self.feature_dim])                           # [batch_size, report_num, report_embedding_dim]
        
        report_representation = torch.cat([title_representation, content_representation, time_representation], dim=2)  # [batch_size, report_num, 3*report_embedding_dim]
        
        # 4. feature fusion
        report_representation = self.feature_fusion(report_representation, category)  # [B, R, 3*feature_dim + cat_dim]

        return report_representation

    
class NAML(ReportEncoder):
    #def __init__(self, config: Config):
    def __init__(self, config):
        super(NAML, self).__init__(config)
        '''
        self.max_time_length = 20
        self.max_content_length = 200
        self.cnn_kernel_num = 64
        self.report_embedding_dim = 64
        self.time_conv = Conv1D('naive', 300, 64, 3)
        self.content_conv = Conv1D('naive', 300, 64, 3)
        self.time_attention = Attention(64, 64)
        self.content_attention = Attention(64, 64)
        self.category_affine = nn.Linear(50, 64, bias=True)
        self.affine1 = nn.Linear(64, 64, bias=True)
        self.affine2 = nn.Linear(64, 1, bias=False)

        '''
        self.max_title_length = config.max_title_length
        self.max_time_length = config.max_time_length
        self.max_content_length = config.max_content_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.report_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.time_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.time_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.time_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)             # batch_size 한 번에 처리하는 샘플 수
        report_num = title_text.size(1)             # report_num 각 샘플 안에 포함된 report(명령) 수
        batch_report_num = batch_size * report_num  # batch_report_num

        # 1. word embedding
        title_emb = self.dropout(self.word_embedding(title_text))       # [batch_size * report_num, max_title_length, word_embedding_dim]
        time_emb  = self.dropout(self.word_embedding(time_text))        # [batch_size * report_num, max_time_length, word_embedding_dim]
        content_emb = self.dropout(self.word_embedding(content_text))   # [batch_size * report_num, max_content_length, word_embedding_dim]

        title_w = title_emb.view(batch_report_num, self.max_title_length, self.word_embedding_dim)         # [batch_size * report_num, max_title_length, word_embedding_dim]
        time_w  = time_emb.view(batch_report_num, self.max_time_length,  self.word_embedding_dim)          # [batch_size * report_num, max_time_length, word_embedding_dim] 
        content_w = content_emb.view(batch_report_num, self.max_content_length, self.word_embedding_dim)    # [batch_size * report_num, max_content_length, word_embedding_dim]

        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                     # [batch_size * report_num, max_title_length, cnn_kernel_num]       
        time_c = self.dropout_(self.time_conv(time_w.permute(0, 2, 1)).permute(0, 2, 1))                                                        # [batch_size * report_num, max_time_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                               # [batch_size * report_num, max_content_length, cnn_kernel_num]

        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, report_num, self.cnn_kernel_num])
        time_representation = self.time_attention(time_c).view([batch_size, report_num, self.cnn_kernel_num])                                   # [batch_size, report_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, report_num, self.cnn_kernel_num])                          # [batch_size, report_num, cnn_kernel_num]

        # 4. category encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                                             # [batch_size, report_num, cnn_kernel_num]
       
        # 5. multi-view attention
        feature = torch.stack([title_representation, time_representation, content_representation, category_representation], dim=2)                                       # [batch_size, report_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                               # [batch_size, report_num, 4, 1]
        report_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                     # [batch_size, report_num, cnn_kernel_num]
        return report_representation
    
class NAML_noTitle(ReportEncoder):
    #def __init__(self, config: Config):
    def __init__(self, config):
        super(NAML_noTitle, self).__init__(config)
        
        self.max_time_length = config.max_time_length
        self.max_content_length = config.max_content_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.report_embedding_dim = config.cnn_kernel_num
        self.time_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        
        self.time_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        

    def initialize(self):
        super().initialize()
        self.time_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)             # batch_size 한 번에 처리하는 샘플 수
        report_num = title_text.size(1)             # report_num 각 샘플 안에 포함된 report(명령) 수
        batch_report_num = batch_size * report_num  # batch_report_num

        # 1. word embedding
        time_emb  = self.dropout(self.word_embedding(time_text))        # [batch_size * report_num, max_time_length, word_embedding_dim]
        content_emb = self.dropout(self.word_embedding(content_text))   # [batch_size * report_num, max_content_length, word_embedding_dim]

        time_w  = time_emb.view(batch_report_num, self.max_time_length,  self.word_embedding_dim)          # [batch_size * report_num, max_time_length, word_embedding_dim] 
        content_w = content_emb.view(batch_report_num, self.max_content_length, self.word_embedding_dim)    # [batch_size * report_num, max_content_length, word_embedding_dim]

        # 2. CNN encoding                                                # [batch_size * report_num, max_title_length, cnn_kernel_num]       
        time_c = self.dropout_(self.time_conv(time_w.permute(0, 2, 1)).permute(0, 2, 1))                                                        # [batch_size * report_num, max_time_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                               # [batch_size * report_num, max_content_length, cnn_kernel_num]

        # 3. attention layer
        time_representation = self.time_attention(time_c).view([batch_size, report_num, self.cnn_kernel_num])                                   # [batch_size, report_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, report_num, self.cnn_kernel_num])                          # [batch_size, report_num, cnn_kernel_num]

        # 4. category encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                                             # [batch_size, report_num, cnn_kernel_num]
       
        # 5. multi-view attention
        feature = torch.stack([time_representation, content_representation, category_representation], dim=2)                                       # [batch_size, report_num, 3, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                               # [batch_size, report_num, 3, 1]
        report_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                     # [batch_size, report_num, cnn_kernel_num]
        return report_representation
    
class NAML_noTime(ReportEncoder):
    def __init__(self, config):
        super(NAML_noTime, self).__init__(config)
        
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_content_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.report_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)             # batch_size 한 번에 처리하는 샘플 수
        report_num = title_text.size(1)             # report_num 각 샘플 안에 포함된 report(명령) 수
        batch_report_num = batch_size * report_num  # batch_report_num

        # 1. word embedding
        title_emb = self.dropout(self.word_embedding(title_text))       # [batch_size * report_num, max_title_length, word_embedding_dim]
        content_emb = self.dropout(self.word_embedding(content_text))   # [batch_size * report_num, max_content_length, word_embedding_dim]

        title_w = title_emb.view(batch_report_num, self.max_title_length, self.word_embedding_dim)         # [batch_size * report_num, max_title_length, word_embedding_dim]
        content_w = content_emb.view(batch_report_num, self.max_content_length, self.word_embedding_dim)    # [batch_size * report_num, max_content_length, word_embedding_dim]

        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                     # [batch_size * report_num, max_title_length, cnn_kernel_num]       
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                               # [batch_size * report_num, max_content_length, cnn_kernel_num]

        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, report_num, self.cnn_kernel_num])
        content_representation = self.content_attention(content_c).view([batch_size, report_num, self.cnn_kernel_num])                          # [batch_size, report_num, cnn_kernel_num]

        # 4. category encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                                             # [batch_size, report_num, cnn_kernel_num]
       
        # 5. multi-view attention
        feature = torch.stack([title_representation, content_representation, category_representation], dim=2)                                       # [batch_size, report_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                               # [batch_size, report_num, 4, 1]
        report_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                     # [batch_size, report_num, cnn_kernel_num]
        return report_representation
     
class NAML_noBody(ReportEncoder):
    def __init__(self, config):
        super(NAML_noBody, self).__init__(config)
        
        self.max_title_length = config.max_title_length
        self.max_time_length = config.max_time_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.report_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.time_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.time_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.time_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)             # batch_size 한 번에 처리하는 샘플 수
        report_num = title_text.size(1)             # report_num 각 샘플 안에 포함된 report(명령) 수
        batch_report_num = batch_size * report_num  # batch_report_num

        # 1. word embedding
        title_emb = self.dropout(self.word_embedding(title_text))       # [batch_size * report_num, max_title_length, word_embedding_dim]
        time_emb  = self.dropout(self.word_embedding(time_text))        # [batch_size * report_num, max_time_length, word_embedding_dim]

        title_w = title_emb.view(batch_report_num, self.max_title_length, self.word_embedding_dim)         # [batch_size * report_num, max_title_length, word_embedding_dim]
        time_w  = time_emb.view(batch_report_num, self.max_time_length,  self.word_embedding_dim)          # [batch_size * report_num, max_time_length, word_embedding_dim] 
        
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                     # [batch_size * report_num, max_title_length, cnn_kernel_num]       
        time_c = self.dropout_(self.time_conv(time_w.permute(0, 2, 1)).permute(0, 2, 1))                                                        # [batch_size * report_num, max_time_length, cnn_kernel_num]
       
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, report_num, self.cnn_kernel_num])
        time_representation = self.time_attention(time_c).view([batch_size, report_num, self.cnn_kernel_num])                                   # [batch_size, report_num, cnn_kernel_num]
        
        # 4. category encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                                             # [batch_size, report_num, cnn_kernel_num]
       
        # 5. multi-view attention
        feature = torch.stack([title_representation, time_representation, category_representation], dim=2)                                       # [batch_size, report_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                               # [batch_size, report_num, 4, 1]
        report_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                     # [batch_size, report_num, cnn_kernel_num]
        return report_representation
    
class NAML_noCategory(ReportEncoder):
    #def __init__(self, config: Config):
    def __init__(self, config):
        super(NAML_noCategory, self).__init__(config)
        
        self.max_title_length = config.max_title_length
        self.max_time_length = config.max_time_length
        self.max_content_length = config.max_content_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.report_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.time_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.time_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.time_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)             # batch_size 한 번에 처리하는 샘플 수
        report_num = title_text.size(1)             # report_num 각 샘플 안에 포함된 report(명령) 수
        batch_report_num = batch_size * report_num  # batch_report_num

        # 1. word embedding
        title_emb = self.dropout(self.word_embedding(title_text))       # [batch_size * report_num, max_title_length, word_embedding_dim]
        time_emb  = self.dropout(self.word_embedding(time_text))        # [batch_size * report_num, max_time_length, word_embedding_dim]
        content_emb = self.dropout(self.word_embedding(content_text))   # [batch_size * report_num, max_content_length, word_embedding_dim]

        title_w = title_emb.view(batch_report_num, self.max_title_length, self.word_embedding_dim)         # [batch_size * report_num, max_title_length, word_embedding_dim]
        time_w  = time_emb.view(batch_report_num, self.max_time_length,  self.word_embedding_dim)          # [batch_size * report_num, max_time_length, word_embedding_dim] 
        content_w = content_emb.view(batch_report_num, self.max_content_length, self.word_embedding_dim)    # [batch_size * report_num, max_content_length, word_embedding_dim]

        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                     # [batch_size * report_num, max_title_length, cnn_kernel_num]       
        time_c = self.dropout_(self.time_conv(time_w.permute(0, 2, 1)).permute(0, 2, 1))                                                        # [batch_size * report_num, max_time_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                               # [batch_size * report_num, max_content_length, cnn_kernel_num]

        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, report_num, self.cnn_kernel_num])
        time_representation = self.time_attention(time_c).view([batch_size, report_num, self.cnn_kernel_num])                                   # [batch_size, report_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, report_num, self.cnn_kernel_num])                          # [batch_size, report_num, cnn_kernel_num]                                           # [batch_size, report_num, cnn_kernel_num]
       
        # 5. multi-view attention
        feature = torch.stack([title_representation, time_representation, content_representation], dim=2)                                       # [batch_size, report_num, 3, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                               # [batch_size, report_num, 3, 1]
        report_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                     # [batch_size, report_num, cnn_kernel_num]
        return report_representation

class NAML_onlyBody(ReportEncoder):
    def __init__(self, config):
        super(NAML_onlyBody, self).__init__(config)

        self.max_content_length = config.max_content_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.report_embedding_dim = config.cnn_kernel_num
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        

    def initialize(self):
        super().initialize()
        self.content_attention.initialize()

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)             # batch_size 한 번에 처리하는 샘플 수
        report_num = title_text.size(1)             # report_num 각 샘플 안에 포함된 report(명령) 수
        batch_report_num = batch_size * report_num  # batch_report_num

        # 1. word embedding
        content_emb = self.dropout(self.word_embedding(content_text))   # [batch_size * report_num, max_content_length, word_embedding_dim]
        content_w = content_emb.view(batch_report_num, self.max_content_length, self.word_embedding_dim)    # [batch_size * report_num, max_content_length, word_embedding_dim]

        # 2. CNN encoding
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                               # [batch_size * report_num, max_content_length, cnn_kernel_num]

        # 3. attention layer
        content_representation = self.content_attention(content_c).view([batch_size, report_num, self.cnn_kernel_num])                          # [batch_size, report_num, cnn_kernel_num]

        report_representation = content_representation                                                                     # [batch_size, report_num, cnn_kernel_num]
        return report_representation

class CNN(ReportEncoder):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_time_length = config.max_time_length
        self.max_content_length = config.max_content_length
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.cnn_kernel_num = config.cnn_kernel_num

        # multi-view attention (3 views)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)

        # 최종 보고서 임베딩 차원
        self.report_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim

    def initialize(self):
        super().initialize()
        self.attention.initialize()
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)             # batch_size: 한 번에 처리하는 샘플 수
        report_num = title_text.size(1)             # report_num: 각 샘플 안에 포함된 report 수
        batch_report_num = batch_size * report_num  # batch_report_num

        # 1. 단어 임베딩
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_report_num, self.max_title_length, self.word_embedding_dim])          # [batch_size * report_num, max_title_length, word_embedding_dim]
        time_w = self.dropout(self.word_embedding(time_text)).view([batch_report_num, self.max_time_length, self.word_embedding_dim])             # [batch_size * report_num, max_time_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_report_num, self.max_content_length, self.word_embedding_dim])    # [batch_size * report_num, max_content_length, word_embedding_dim]

        # 2. CNN 인코딩
        title_c = self.dropout_(self.conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                # [batch_size * report_num, max_title_length, cnn_kernel_num]
        time_c = self.dropout_(self.conv(time_w.permute(0, 2, 1)).permute(0, 2, 1))                  # [batch_size * report_num, max_time_length, cnn_kernel_num]
        content_c = self.dropout_(self.conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))         # [batch_size * report_num, max_content_length, cnn_kernel_num]

        # 3. 어텐션 레이어
        title_representation = self.attention(title_c, mask=title_mask.view(batch_report_num, self.max_title_length)).view(batch_size, report_num, self.cnn_kernel_num)
        time_representation = self.attention(time_c, mask=time_mask.view(batch_report_num, self.max_time_length)).view(batch_size, report_num, self.cnn_kernel_num)
        content_representation = self.attention(content_c, mask=content_mask.view(batch_report_num, self.max_content_length)).view(batch_size, report_num, self.cnn_kernel_num)

        # 4) multi-view attention fuse (3 -> 1): [B, R, K]
        feature = torch.stack([title_representation, time_representation, content_representation], dim=2)           # [B, R, 3, K]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [B, R, 3, 1]
        text_fused = (feature * alpha).sum(dim=2)                                  # [B, R, K]

        # 5) feature fusion (category concat): [B, R, K + cat_dim]
        report_representation = self.feature_fusion(text_fused, category)

        return report_representation
    
class CROWN(ReportEncoder):
    def __init__(self, config):
        super(CROWN, self).__init__(config)

        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_content_length 
        self.category_embedding_dim = config.category_embedding_dim
        self.intent_embedding_dim = config.intent_embedding_dim
        
        self.report_embedding_dim = config.intent_embedding_dim * 2 + config.category_embedding_dim 
        
        # Transformer encoder
        self.title_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_title_length)
        self.content_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_content_length)
        title_encoder_layers = TransformerEncoderLayer(config.word_embedding_dim, config.head_num, config.feedforward_dim, config.dropout_rate, batch_first=True)   
        self.title_transformer = TransformerEncoder(title_encoder_layers, config.num_layers)
        content_encoder_layers = TransformerEncoderLayer(config.word_embedding_dim, config.head_num, config.feedforward_dim, config.dropout_rate, batch_first=True)  
        self.content_transformer = TransformerEncoder(content_encoder_layers, config.num_layers)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.category_embedding_dim, bias=True)
        # ISAB(Induced Set Attention Block) emcoder
        self.ISAB = ISAB(dim_in = config.word_embedding_dim, 
                         dim_out = config.word_embedding_dim,
                         num_heads = config.isab_num_heads,        # The number of ISAB heads       4,  choices=[2, 4, 8]
                         num_inds = config.isab_num_inds,          # The number of inducing points  4,  choices=[2, 4, 6, 8]
                         ln = True)
        
        
        self.intent_num = config.intent_num     # hyperparameter k
        self.alpha = config.alpha 
        self.title_intent_attention = Attention(config.intent_embedding_dim, config.attention_dim) 
        self.content_intent_attention = Attention(config.intent_embedding_dim, config.attention_dim)
        self.intent_layers = nn.ModuleList([nn.Linear(config.word_embedding_dim
                                                      + config.category_embedding_dim
                                                      , config.intent_embedding_dim, bias=True) 
                                            for _ in range(self.intent_num)])
        
        self.category_predictor = CategoryPredictor(config.intent_embedding_dim, config.category_num)

    
    def initialize(self):
        super().initialize()
        self.title_intent_attention.initialize()
        self.content_intent_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        # Initialize each intent layer with different weights to learn different embedding for each intent
        for intent_layer in self.intent_layers:
            nn.init.xavier_uniform_(intent_layer.weight)
            nn.init.zeros_(intent_layer.bias)

    # Apply k-FC layer for k-intent disentanglement
    def k_intent_disentangle(self, intent_num, report_embedding):                                        
        k_intent_embeddings = []
        for i in range(intent_num):
            # Apply different linear transformations for each intent
            intent_embedding = F.relu(self.intent_layers[i](report_embedding), inplace=True)      # [batch_size * report_num, intent_embedding_dim]     
            # Expand the dimension (axis 1)
            intent_embedding_exp = intent_embedding.unsqueeze(1)
            k_intent_embeddings.append(intent_embedding_exp)
        # Concatenate the k_intent_embeddings along the second axis (axis 1)
        k_intent_embeddings = torch.cat(k_intent_embeddings, dim=1)                             # [batch_size * report_num, intent_length, intent_embedding_dim]

        return k_intent_embeddings

    def similarity_compute(self, title, content):                              # [batch_size * report_num, intent_embedding_dim]
        cosine_similarity = F.cosine_similarity(title, content, dim=1)             
        title_content_similarity = (cosine_similarity + 1) / 2.0
        return title_content_similarity    
    
    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = title_text.size(0)
        report_num = title_text.size(1)
        batch_report_num = batch_size * report_num
        
        t_mask = title_mask.view([batch_report_num, self.max_title_length])                                   # [batch_size * report_num, max_title_length]
        b_mask = content_mask.view([batch_report_num, self.max_content_length])                                  # [batch_size * report_num, max_content_length]
        
        # Word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_report_num, self.max_title_length, self.word_embedding_dim])          # [batch_size * report_num, max_title_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_report_num, self.max_content_length, self.word_embedding_dim])          # [batch_size * report_num, max_content_length, word_embedding_dim]
        
        # Transformer encoding
        title_p = self.title_pos_encoder(title_w)                                                       # [batch_size * report_num, max_title_length, report_embedding_dim]
        title_t = self.title_transformer(title_p)                                                       # [batch_size * report_num, max_title_length, report_embedding_dim]
        title_embedding = title_t.mean(dim=1).view([batch_size * report_num, self.word_embedding_dim])    # [batch_size * report_num, report_embedding_dim]    
        
        content_p = self.content_pos_encoder(content_w)                                                          # [batch_size * report_num, max_content_length, report_embedding_dim]
        content_t = self.content_transformer(content_p)                                                          # [batch_size * report_num, max_content_length, report_embedding_dim]
        content_embedding = content_t.mean(dim=1).view([batch_size * report_num, self.word_embedding_dim])      # [batch_size * report_num, report_embedding_dim]   

        # Category-aware intent disentanglement
        category_representation = self.category_affine(self.category_embedding(category)).view([batch_report_num, self.category_embedding_dim])   # [batch_size * report_num, category_embedding_dim] 
        category_aware_title_embedding = torch.cat([title_embedding, category_representation], dim=1)                            # [batch_size * report_num, report_embedding_dim + category_embedding_dim]
        category_aware_content_embedding = torch.cat([content_embedding, category_representation], dim=1)                             # [batch_size * report_num, report_embedding_dim + category_embedding_dim]

        k = self.intent_num
        title_k_intent_embeddings = self.k_intent_disentangle(k, category_aware_title_embedding)              # [batch_size * report_num, intent_length(k), intent_embedding_dim]
        content_k_intent_embeddings = self.k_intent_disentangle(k, category_aware_content_embedding)                # [batch_size * report_num, intent_length(k), intent_embedding_dim]
        
        # Intent-based Attention
        title_intent_embedding = self.title_intent_attention(title_k_intent_embeddings)              # [batch_size * report_num, intent_embedding_dim]  [batch_size * report_num, 1, intent_length(k)]
        content_intent_embedding = self.content_intent_attention(content_k_intent_embeddings)                  # [batch_size * report_num, intent_embedding_dim]  [batch_size * report_num, 1, intent_length(k)]
        
        # Category predictor
        # title_embedding     : [batch_size * report_num, intent_embedding_dim]
        # target category     : [batch_size * report_num, 1]
        target_category = category.view([batch_report_num, 1])
        category_loss = self.category_predictor(title_intent_embedding, target_category, self.category_num)
        
        self.auxiliary_loss = category_loss * self.alpha
        
        # Title-content similarity computation
        title_content_similarity = self.similarity_compute(title_intent_embedding, content_intent_embedding).view([batch_size * report_num, 1])  # [batch_size * report_num, 1]
        
        report_representation = torch.cat([title_intent_embedding, title_content_similarity * content_intent_embedding], dim=1).view([batch_size, report_num, self.intent_embedding_dim * 2])       # [batch_size, report_num, intent_embedding_dim * 2]         # concat 
        # report_representation = (title_intent_embedding + title_content_similarity * content_intent_embedding).view([batch_size, report_num, self.intent_embedding_dim])                          # [batch_size, report_num, title(content) intent_embedding_dim]     # average(weighted sum)
        report_representation = self.feature_fusion(report_representation, category)                                   # [batch_size, report_num, intent_embedding_dim + 100]

        return report_representation                                                                                              # [batch_size, report_num, report_embedding_dim]    

class CategoryPredictor(nn.Module):
    def __init__(self, title_embedding, category_num):
        super(CategoryPredictor, self).__init__()
        self.fc = nn.Linear(title_embedding, category_num)

    def one_hot_encode(self, target, category_num):
        batch_size, _ = target.size()
        one_hot = torch.zeros(batch_size, category_num, device=target.device)
        target = target.long()
        one_hot.scatter_(1, target, 1)
        return one_hot
    # Input: title_intent_embedding             # [batch_size * report_num, intent_embedding_dim]
    # Output: category loss (auxiliary loss)    # [batch_size, report_num]
    def forward(self, title_intent_embedding, targets, category_num):
        category_logits = self.fc(title_intent_embedding)               # [batch_size * report_num, category_num]
        one_hot_targets = self.one_hot_encode(targets, category_num)    # [batch_size * report_num, category_num]
        category_loss = F.cross_entropy(category_logits, one_hot_targets)

        return category_loss

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
    

'''
LIME - User-Topic Lifetime-aware Age Encoder

Input:
 - age:                 [batch_size, report_num] 
 - user_topic_lifetime: [batch_size, report_num]
Algorithm:
(1) log-scaling + bucketization
(2) Concat → Dense layer → Tanh
'''

class FreshnessEncoder(nn.Module):
    def __init__(self, config, base_report_encoder: nn.Module):
        super(FreshnessEncoder, self).__init__()
        embedding_dim=config.freshness_embedding_dim
        if config.fusion_method == 'add' or 'gated':
            hidden_dim= base_report_encoder.report_embedding_dim
        else:
            hidden_dim= config.lime_hidden_dim
        self.num_buckets = config.num_buckets
        self.freshness_embedding = nn.Embedding(self.num_buckets, embedding_dim)
        self.lifetime_embedding = nn.Embedding(self.num_buckets, embedding_dim)

        self.dense = nn.Linear(embedding_dim * 2, hidden_dim)
        self.activation = nn.Tanh()

    def bucketize(self, x):
        x = torch.clamp(x.float(), min=1)  
        log_x = torch.log(x)
        denom = torch.log(torch.tensor(60 * 60 * 24.0, device=x.device))
        scaled = log_x / denom
        buckets = torch.clamp((scaled * (self.num_buckets / 7)).long(), max=self.num_buckets - 1)
        return buckets

    def forward(self, report_freshness, report_user_topic_lifetime):
        """
        Input
        freshness:              [batch_size, report_num], unit = sec
        user_topic_lifetime:    [batch_size, report_num], unit = sec
        """

        if report_freshness.dim() == 1:
            report_freshness = report_freshness.unsqueeze(1)  # [B] → [B, 1]
        if report_user_topic_lifetime.dim() == 1:
            report_user_topic_lifetime = report_user_topic_lifetime.unsqueeze(1)

        if report_freshness.shape != report_user_topic_lifetime.shape:
            report_user_topic_lifetime = report_user_topic_lifetime.expand_as(report_freshness)
        
        f_bucket = self.bucketize(report_freshness)
        l_bucket = self.bucketize(report_user_topic_lifetime)

        f_embed = self.freshness_embedding(f_bucket)    # [batch_size, report_num, report_embedding_dim]
        l_embed = self.lifetime_embedding(l_bucket)     # [batch_size, report_num, report_embedding_dim]

        concat = torch.cat([f_embed, l_embed], dim=-1)  # [batch_size, report_num, report_embedding_dim * 2]
        output = self.activation(self.dense(concat))    # [batch_size, report_num, hidden_dim]
        return output



class LIME(nn.Module):
    def __init__(self, config, base_report_encoder: nn.Module):
        super(LIME, self).__init__()
        self.final_dim = config.lime_output_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        self.category_embedding.weight.requires_grad = False
        self.subCategory_embedding = nn.Embedding(config.subCategory_num, config.subCategory_embedding_dim)
        self.subCategory_embedding.weight.requires_grad = False
        self.category_affine = nn.Linear(config.category_embedding_dim + config.subCategory_embedding_dim, config.category_embedding_dim)
        
        self.base_report_encoder = base_report_encoder  
        self.freshness_encoder = FreshnessEncoder(config, base_report_encoder)
        self.fusion_method = config.fusion_method       # 'concat' or 'add' or 'gated'
        if hasattr(base_report_encoder, "auxiliary_loss"):
            self.auxiliary_loss = base_report_encoder.auxiliary_loss
        else:
            self.auxiliary_loss = None

        content_dim = self.base_report_encoder.report_embedding_dim
        if self.fusion_method == 'add' or 'gated':
            freshness_dim = content_dim
        else:
            freshness_dim = config.lime_hidden_dim

        if self.fusion_method == 'concat':
            self.output_dim = content_dim + freshness_dim
            if self.final_dim:
                self.project = nn.Linear(self.output_dim, self.final_dim)
                self.output_dim = self.final_dim
            else:
                self.project = nn.Identity()
        elif self.fusion_method == 'add':
            assert content_dim == freshness_dim, "Add fusion requires same dim"
            self.output_dim = content_dim
            self.project = nn.Identity()
        elif self.fusion_method == 'gated':
            self.gate = nn.Linear(content_dim + freshness_dim, content_dim)
            self.output_dim = content_dim
            self.project = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        self.report_embedding_dim = self.output_dim

    def initialize(self):
        if hasattr(self.base_report_encoder, 'initialize'):
            self.base_report_encoder.initialize()
        nn.init.xavier_uniform_(self.freshness_encoder.dense.weight)
        nn.init.zeros_(self.freshness_encoder.dense.bias)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
    
    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding, report_freshness, report_user_topic_lifetime):
        """
        report_freshness, report_user_topic_lifetime: [batch_size, report_num]
        
        """
        content_emb = self.base_report_encoder(title_text, title_mask, content_text, content_mask, time_text, time_mask,
                                            category, user_embedding, report_freshness, report_user_topic_lifetime)  # [B, N, D_c]
        freshness_emb = self.freshness_encoder(report_freshness, report_user_topic_lifetime)  # [B, N, D_f]
        if freshness_emb.dim() == 2:
            freshness_emb = freshness_emb.unsqueeze(1)

        if self.fusion_method == 'concat':
            combined = torch.cat([content_emb, freshness_emb], dim=-1)
            output = self.project(combined)
        elif self.fusion_method == 'add':
            output = content_emb + freshness_emb
        elif self.fusion_method == 'gated':
            gate_input = torch.cat([content_emb, freshness_emb], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            output = gate * content_emb + (1 - gate) * freshness_emb
        
        return output  # [B, N, output_dim]