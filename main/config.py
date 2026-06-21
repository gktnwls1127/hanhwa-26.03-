from html import parser
import os
import argparse
import sys
import time
import torch
import random
import numpy as np
import json
from prepare_Command_Dataset import prepare_command_dataset
from prepare_Command_TimeDataset import prepare_command_time_dataset
from prepare_Command_UnitDataset import prepare_command_unit_dataset

class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='Neural report recommendation')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--report_encoder', type=str, default='NAML', choices=['CNN', 'MHSA', 'NAML', 'NAML_noTitle', 'NAML_noTime', 'NAML_noBody', 'NAML_noCategory', 'NAML_onlyBody', 'CROWN', 'LIME'], help='Report encoder')
        parser.add_argument('--user_encoder', type=str, default='ATT', choices=['LSTUR', 'MHSA', 'ATT', 'ATT_noPosition', 'CROWN'], help='User encoder')
        # 공유기 추가(26.06)
        parser.add_argument('--unit_encoder', type=str, default='ATT', choices=['ATT', 'ATT_noName', 'ATT_noType', 'ATT_noNameType', 'CROWN'], help='Unit encoder')
        # LIME 추가(26.05)
        parser.add_argument('--content_encoder', type=str, default='CROWN', choices=['CROWN', 'NAML'], help='Base report content encoder used inside LIME')
        
        parser.add_argument('--dev_model_path', type=str, default='', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='', help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Specific test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--config_file', type=str, default='', help='Config file path')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='small', choices=['small', 'large', 'Jan', 'March', 'April', 'May', 'unit'], help='Dataset type')
        parser.add_argument('--tokenizer', type=str, default='SentencePiece', choices=['Command', 'Mecab', 'SentencePiece', 'NLTK'], help='Sentence tokenizer')
        parser.add_argument('--word_threshold', type=int, default=3, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=32, help='Sentence truncate length for title')
        parser.add_argument('--max_content_length', type=int, default=128, help='Sentence truncate length for content')
        parser.add_argument('--max_time_length', type=int, default=32, help='Sentence truncate length for time')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
        parser.add_argument('--max_history_num', type=int, default=50, help='Maximum number of history reports for each user')
        parser.add_argument('--epoch', type=int, default=10, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=4, help='Gradient clip norm (non-positive value for no clipping)')
        parser.add_argument('--world_size', type=int, default=1, help='World size of multi-process GPU training')
        
        # epoch test 여부 추가
        parser.add_argument('--epoch_test', action='store_true', help='Evaluate test set at every epoch')
        # 시간별 test 여부 추가
        parser.add_argument('--time_eval', action='store_true', help='Use temporal evaluation dataset')
        parser.add_argument('--time_train_idx', type=int, default=1, choices=range(1, 10), help='Temporal train index: 1~9')

        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='avg', choices=['auc', 'mrr', 'ndcg5', 'ndcg10', 'avg'], help='Validation criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=16, help='Epoch number of stop training after dev result does not improve')
        
        # LIME config
        parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'add', 'gated'], help='Fusion method between content and freshness embeddings in LIME')
        parser.add_argument('--freshness_embedding_dim', type=int, default=500, help='Embedding dimension of freshness and lifetime in LIME')
        parser.add_argument('--lime_hidden_dim', type=int, default=200, help='Hidden dimension inside freshness encoder')
        parser.add_argument('--lime_output_dim', type=int, default=400, help='Final output dim of LIME ReportEncoder')
        parser.add_argument('--num_buckets', type=int, default=10, help='Num_buckets for Lifetime-aware Freshness Encoder')
        parser.add_argument('--use_candidate_aware_clicked_report_attention', type=bool, default=True, choices=[True, False], help='Use Candidate-aware Clicked Report Attention in LIME')
        parser.add_argument('--use_residual_connection', type=bool, default=True, choices=[True, False], help='Use residual connection in candidate-aware attention in LIME')
        parser.add_argument('--lifetime_type', type=str, default='user_topic', choices=['fixed', 'topic_wise', 'user_topic'], help='Definition of lifetime: fixed/topic_wise/user_topic')
        parser.add_argument('--use_remaining_lifetime_weighting', type=bool, default=True, choices=[True, False], help='Use remaining lifetime guided weighting in LIME')
        parser.add_argument('--sigmoid_scaling_alpha', type=float, default=0.3, help='Scaling factor alpha for sigmoid weighting function in LANCER/LIME')
        parser.add_argument('--penalty_scaling_beta', type=float, default=0.3, help='Penalty factor applied when remaining lifetime is negative in LIME')
        parser.add_argument('--use_expired_penalty', type=bool, default=True, choices=[True, False], help='Apply penalty when remaining lifetime is negative in LIME')
        parser.add_argument('--fixed_lifetime', type=int, default=36*3600, help='Fixed lifetime value(seconds) for lifetime_type=fixed in LANCER')
        
        # Model config
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--context_embedding_dim', type=int, default=100, choices=[100], help='Context embedding dimension')
        parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'], help='CNN group')
        parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernel')
        parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size of CNN kernel')
        parser.add_argument('--attention_dim', type=int, default=200, help="Attention dimension")
        parser.add_argument('--head_num', type=int, default=20, help='Head number of multi-head self-attention')
        parser.add_argument('--head_dim', type=int, default=20, help='Head dimension of multi-head self-attention')
        parser.add_argument('--user_embedding_dim', type=int, default=50, help='User embedding dimension')
        parser.add_argument('--category_embedding_dim', type=int, default=50, help='Category embedding dimension')
        parser.add_argument('--position_embedding_dim', type=int, default=50, help='Position embedding dimension')
        # 공유기 추가
        parser.add_argument('--unit_name_embedding_dim', type=int, default=50)
        parser.add_argument('--unit_type_embedding_dim', type=int, default=50)
        # CRWON 추가 속성
        parser.add_argument('--intent_embedding_dim', type=int, default=400, choices=[100, 200, 300, 400], help='Intent embedding dimension')
        parser.add_argument('--intent_num', type=int, default=3, choices=[1, 2, 3, 4, 5], help='The number of title/body intent (k)')
        parser.add_argument('--isab_num_inds', type=int, default=4, choices=[2, 4, 6, 8, 10], help='The number of inducing points')
        parser.add_argument('--isab_num_heads', type=int, default=4, choices=[2, 4, 6, 10], help='The number of ISAB heads')
        parser.add_argument('--feedforward_dim', type=int, default=512, choices=[128, 256, 512, 1024], help="The dimension of the feedforward network model")
        parser.add_argument('--num_layers', type=int, default=1, choices=[1, 2], help="The number of sub-encoder-layers in transformer encoder")
        parser.add_argument('--alpha', type=float, default=0.3, help='Loss weight for category predictor')

        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--no_self_connection', default=False, action='store_true', help='Whether the graph contains self-connection')
        parser.add_argument('--no_adjacent_normalization', default=False, action='store_true', help='Whether normalize the adjacent matrix')
        parser.add_argument('--gcn_normalization_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric'], help='GCN normalization for adjacent matrix A (\"symmetric\" for D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}; \"asymmetric\" for D^{-\\frac{1}{2}}A)')
        parser.add_argument('--gcn_layer_num', type=int, default=4, help='Number of GCN layer')
        parser.add_argument('--no_gcn_residual', default=False, action='store_true', help='Whether apply residual connection to GCN')
        parser.add_argument('--gcn_layer_norm', default=False, action='store_true', help='Whether apply layer normalization to GCN')
        parser.add_argument('--hidden_dim', type=int, default=200, help='Encoder hidden dimension')
        parser.add_argument('--Alpha', type=float, default=0.1, help='Reconstruction loss weight for DAE')
        parser.add_argument('--long_term_masking_probability', type=float, default=0.1, help='Probability of masking long-term representation for LSTUR')
        parser.add_argument('--personalized_embedding_dim', type=int, default=200, help='Personalized embedding dimension for NPA')
        parser.add_argument('--HDC_window_size', type=int, default=3, help='Convolution window size of HDC for FIM')
        parser.add_argument('--HDC_filter_num', type=int, default=150, help='Convolution filter num of HDC for FIM')
        parser.add_argument('--conv3D_filter_num_first', type=int, default=32, help='3D matching convolution filter num of the first layer for FIM ')
        parser.add_argument('--conv3D_kernel_size_first', type=int, default=3, help='3D matching convolution kernel size of the first layer for FIM')
        parser.add_argument('--conv3D_filter_num_second', type=int, default=16, help='3D matching convolution filter num of the second layer for FIM ')
        parser.add_argument('--conv3D_kernel_size_second', type=int, default=3, help='3D matching convolution kernel size of the second layer for FIM')
        parser.add_argument('--maxpooling3D_size', type=int, default=3, help='3D matching pooling size for FIM ')
        parser.add_argument('--maxpooling3D_stride', type=int, default=3, help='3D matching pooling stride for FIM')
        parser.add_argument('--OMAP_head_num', type=int, default=3, help='Head num of OMAP for Hi-Fi Ark')
        parser.add_argument('--HiFi_Ark_regularizer_coefficient', type=float, default=0.1, help='Coefficient of regularization loss for Hi-Fi Ark')
        parser.add_argument('--click_predictor', type=str, default='dot_product', choices=['dot_product', 'mlp', 'sigmoid', 'FIM'], help='Click predictor')

        args = parser.parse_args()

        # 공유기(unit) 모드 여부(26.06)
        self.unit_eval = (
            args.dataset == 'unit'
            or any(arg == '--unit_encoder' or arg.startswith('--unit_encoder=') for arg in sys.argv)
        )

        self.attribute_dict = dict(vars(args))
        self.attribute_dict['unit_eval'] = self.unit_eval

        for attribute in self.attribute_dict:
            setattr(self, attribute, self.attribute_dict[attribute])
        # 시간별 테스트 추가
        if self.time_eval:
            self.train_root = '../Command-%s/time/train_%d' % (self.dataset, self.time_train_idx)
            self.dev_root = '../Command-%s/time/dev' % self.dataset
            self.test_root = '../Command-%s/time/test' % self.dataset
        # 공유기 추가(26.06)
        elif self.unit_eval:
            self.train_root = '../Command-%s/train' % self.dataset
            self.dev_root   = '../Command-%s/dev' % self.dataset
            self.test_root  = '../Command-%s/test' % self.dataset
        else:
            self.train_root = '../Command-%s/train' % self.dataset
            self.dev_root   = '../Command-%s/dev' % self.dataset
            self.test_root  = '../Command-%s/test' % self.dataset
        self.dropout_rate = 0.25
        self.gcn_layer_num = 3
        '''
        if self.dataset == 'small': # suggested configuration for Command-small
            self.dropout_rate = 0.25
            self.gcn_layer_num = 3
        else: # suggested configuration for Command-large
            self.dropout_rate = 0.25
            self.gcn_layer_num = 3
        '''
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())
        self.attribute_dict['dropout_rate'] = self.dropout_rate
        self.attribute_dict['gcn_layer_num'] = self.gcn_layer_num
        self.attribute_dict['epoch'] = self.epoch
        self.attribute_dict['seed'] = self.seed
        if self.config_file != '':
            if os.path.exists(self.config_file):
                print('Get experiment settings from the config file : ' + self.config_file)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    for attribute in self.attribute_dict:
                        if attribute in configs:
                            setattr(self, attribute, configs[attribute])
                            self.attribute_dict[attribute] = configs[attribute]
            else:
                raise Exception('Config file does not exist : ' + self.config_file)
        assert not (self.no_self_connection and not self.no_adjacent_normalization), 'Adjacent normalization of graph only can be set in case of self-connection'
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for attribute in self.attribute_dict:
            print(attribute + ' : ' + str(getattr(self, attribute)))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        assert self.batch_size % self.world_size == 0, 'For multi-gpu training, batch size must be divisible by world size'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1024'


    def set_cuda(self):
        self.gpu_available = torch.cuda.is_available()
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.gpu_available:
            torch.cuda.set_device(self.device_id)
            torch.cuda.manual_seed(self.seed)
        else:
            print('GPU is not available. Running on CPU.')


    def preliminary_setup(self):
        # 공유기 추가
        if self.unit_eval:
            dataset_files = [
                os.path.join(self.train_root, 'units.tsv'),
                os.path.join(self.train_root, 'commands.tsv'),
                os.path.join(self.train_root, 'behaviors.tsv'),
                os.path.join(self.dev_root, 'units.tsv'),
                os.path.join(self.dev_root, 'commands.tsv'),
                os.path.join(self.dev_root, 'behaviors.tsv'),
                os.path.join(self.test_root, 'units.tsv'),
                os.path.join(self.test_root, 'commands.tsv'),
                os.path.join(self.test_root, 'behaviors.tsv'),
            ]
        else:
            dataset_files = [
                os.path.join(self.train_root, 'users.tsv'),
                os.path.join(self.train_root, 'commands.tsv'),
                os.path.join(self.train_root, 'behaviors.tsv'),
                os.path.join(self.dev_root, 'users.tsv'),
                os.path.join(self.dev_root, 'commands.tsv'),
                os.path.join(self.dev_root, 'behaviors.tsv'),
                os.path.join(self.test_root, 'users.tsv'),
                os.path.join(self.test_root, 'commands.tsv'),
                os.path.join(self.test_root, 'behaviors.tsv'),
            ]
        
        if not all(list(map(os.path.exists, dataset_files))):
            # 시간별 테스트 추가
            if self.time_eval:
                prepare_command_time_dataset(out_dir='../Command-%s/time' % self.dataset)
            # 공유기 추가(26.06)
            elif self.unit_eval:
                prepare_command_unit_dataset(out_dir='../Command-%s' % self.dataset)
            else:
                prepare_command_dataset(out_dir='../Command-%s' % self.dataset)

        if self.report_encoder == 'LIME':
            # 공유기 추가(26.06)
            if self.unit_eval:
                model_name = self.report_encoder + '-' + self.content_encoder + '-' + self.unit_encoder
            else:
                model_name = self.report_encoder + '-' + self.content_encoder + '-' + self.user_encoder
        else:
            # 공유기 추가(26.06)    
            if self.unit_eval:
                model_name = self.report_encoder + '-' + self.unit_encoder
            else:
                model_name = self.report_encoder + '-' + self.user_encoder
        # model_name = self.report_encoder + '-' + self.user_encoder
        mkdirs = lambda x: os.makedirs(x) if not os.path.exists(x) else None
        self.config_dir = 'configs/' + self.dataset + '/' + model_name
        self.model_dir = 'models/' + self.dataset + '/' + model_name
        self.best_model_dir = 'best_model/' + self.dataset + '/' + model_name
        self.dev_res_dir = 'dev/res/' + self.dataset + '/' + model_name
        self.test_res_dir = 'test/res/' + self.dataset + '/' + model_name
        self.result_dir = 'results/' + self.dataset + '/' + model_name
        mkdirs(self.config_dir)
        mkdirs(self.model_dir)
        mkdirs(self.best_model_dir)
        mkdirs('dev/ref')
        mkdirs(self.dev_res_dir)
        mkdirs('test/ref')
        mkdirs(self.test_res_dir)
        mkdirs(self.result_dir)

        # 시간별 평가용 truth 파일 생성
        truth_suffix = '-time' if self.time_eval else ''

        def _extract_label(impression: str):
            """behaviors.tsv에서 user-label 쌍 파싱
            형식: userId-label (e.g., user_001-1, user_002-0)
            """
            impression = impression.strip()
            if not impression:
                return None
            if impression.endswith('-1'):
                return 1
            if impression.endswith('-0'):
                return 0
            try:
                label = impression.rsplit('-', 1)[-1]
                if label == '1':
                    return 1
                if label == '0':
                    return 0
            except Exception:
                return None
            return None

        # Dev set 평가용 truth 파일 생성 (behaviors.tsv 기반)
        # 시간별 평가 dev truth 파일
        dev_truth_path = 'dev/ref/truth-%s%s.txt' % (self.dataset, truth_suffix)

        if not os.path.exists(dev_truth_path):
            behaviors_tsv = os.path.join(self.dev_root, 'behaviors.tsv')
            with open(behaviors_tsv, 'r', encoding='utf-8') as dev_f:
                with open(dev_truth_path, 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        cols = line.rstrip('\n').split('\t')
                        # behaviors.tsv 형식: ImpressionID, CommandID, ReportTime, Impressions
                        impressions = cols[3] if len(cols) > 3 else ''
                        labels = []
                        for impression in impressions.strip().split(' '):
                            label_val = _extract_label(impression)
                            if label_val is None:
                                continue
                            labels.append(label_val)
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        
        # Test set 평가용 truth 파일 생성 (behaviors.tsv 기반)
        # 시간별 평가 test truth 파일
        test_truth_path = 'test/ref/truth-%s%s.txt' % (self.dataset, truth_suffix)
        if not os.path.exists(test_truth_path):
            behaviors_tsv = os.path.join(self.test_root, 'behaviors.tsv')
            with open(behaviors_tsv, 'r', encoding='utf-8') as test_f:
                with open(test_truth_path, 'w', encoding='utf-8') as truth_f:
                    for test_ID, line in enumerate(test_f):
                        cols = line.rstrip('\n').split('\t')
                        impressions = cols[3] if len(cols) > 3 else ''
                        labels = []
                        for impression in impressions.strip().split(' '):
                            label_val = _extract_label(impression)
                            if label_val is None:
                                continue
                            labels.append(label_val)
                        truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))
        else:
            self.prediction_dir = 'prediction/large/' + model_name
            mkdirs(self.prediction_dir)


    def __init__(self):
        self.parse_argument()
        self.preliminary_setup()
        self.set_cuda()


if __name__ == '__main__':
    config = Config()
