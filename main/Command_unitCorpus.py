import os
import json
import pickle
import collections
import re
import sys
import csv
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe
from config import Config
import torch
import numpy as np
from konlpy.tag import Mecab
from gensim.models import FastText
from datetime import datetime
import bisect
try:
    import sentencepiece as spm
except Exception:
    spm = None
try:
    mecab = Mecab()
except Exception:
    mecab = None



def _parse_time_to_ts(s: str) -> int:
    if s is None:
        return 0
    s = str(s).strip()
    if not s:
        return 0
    try:
        # Python 3.7+ : 'YYYY-MM-DDTHH:MM:SS+09:00' 지원
        dt = datetime.fromisoformat(s)
        # timezone-aware면 timestamp OK
        return int(dt.timestamp())
    except Exception:
        # 혹시 형식이 깨진 경우 대비(최소)
        try:
            # 예: 'YYYY-MM-DD HH:MM:SS' 같은 케이스
            dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
            return int(dt.timestamp())
        except Exception:
            return 0


def _set_csv_field_limit():
    max_int = getattr(sys, "maxsize", 2**31 - 1)
    for v in [max_int, 2**31 - 1, 10_000_000, 1_000_000]:
        try:
            csv.field_size_limit(v)
            return
        except Exception:
            continue

_set_csv_field_limit()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

_pat_fallback = re.compile(r"[0-9A-Za-z가-힣]+|[^\s]")

def safe_word_tokenize(text, tokenizer_type='word_tokenize', sp_proc=None):
    if tokenizer_type == 'MeCab' and mecab:
        return mecab.morphs(text)
    elif tokenizer_type == 'SentencePiece' and sp_proc:
        return sp_proc.encode(text, out_type=str)
    try:
        return word_tokenize(text)
    except Exception:
        return _pat_fallback.findall(text)


class Command_unitCorpus:
    @staticmethod
    def _load_commands(path):
        commands = []
        if not os.path.exists(path):
            return commands
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quotechar='"', escapechar='\\')
            for row in reader:
                if len(row) < 7:
                    continue
                commands.append(
                    {
                        "dataId": row[0],
                        "validUntil": row[1],
                        "securityLevel": row[2],
                        "title": row[3],
                        "reportTime": row[4],
                        "body": row[5],
                        "category": row[6],
                    }
                )
        return commands

    @staticmethod
    def _load_units(path):
        units = []
        if not os.path.exists(path):
            return units

        import sys
        try:
            csv.field_size_limit(sys.maxsize)
        except Exception:
            pass

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quotechar='"', escapechar='\\')

            for row in reader:
                if len(row) < 7:
                    continue

                units.append(
                    {
                        "unitId": row[0],
                        "name": row[1],
                        "unitSize": row[2],
                        "unitType": row[3],
                        "combatPower": row[4],
                        "location": row[5],
                        "history": row[6],
                    }
                )

        return units
    
    @staticmethod
    def _load_behaviors(path):
        """behaviors.tsv 로드 (UNIT 추천 방식)
        columns: ImpressionID, CommandID, ReportTime, Impressions
        Impressions: unitId1-label1 unitId2-label2 ... (label=1이면 해당 unit이 명령을 읽음)
        """
        behaviors = []
        if not os.path.exists(path):
            return behaviors
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quotechar='"', escapechar='\\')
            for row in reader:
                if len(row) < 4:
                    continue
                behaviors.append(
                    {
                        "ImpressionID": row[0],
                        "CommandID": row[1],
                        "ReportTime": row[2],
                        "Impressions": row[3],
                    }
                )
        return behaviors

    @staticmethod
    def _history_from_str(history_str):
        if history_str is None:
            return []
        return [h.strip() for h in history_str.split(" ") if h.strip()]

    @staticmethod
    def _parse_time_to_ts(s: str) -> int:
        if s is None:
            return 0
        s = str(s).strip()
        if not s:
            return 0
        try:
            # '2025-10-30T06:33:00+09:00' 같은 ISO8601 처리
            dt = datetime.fromisoformat(s)
            return int(dt.timestamp())
        except Exception:
            return 0

    @staticmethod
    def _build_unitidx_to_graphrow(order, unit_ID_dict):
        mapping = {}
        for i, uid in enumerate(order):
            uid = "" if uid is None else str(uid).strip()
            if uid == "":
                continue
            mapping[unit_ID_dict.get(uid, 0)] = i
        return mapping

    @staticmethod
    def preprocess(config: Config):
        # 26.05 추가
        cache_dataset = config.dataset

        if getattr(config, "time_eval", False):
            cache_dir = os.path.join("cache", "time", config.dataset, "all")
            cache_dataset = f"{config.dataset}-time"
        else:
            cache_dir = os.path.join("cache", "unit", config.dataset)
            cache_dataset = f"{config.dataset}-unit"

        os.makedirs(cache_dir, exist_ok=True)

        unit_ID_file = os.path.join(cache_dir, 'unit_ID-%s.json' % cache_dataset)
        report_ID_file = os.path.join(cache_dir, 'report_ID-%s.json' % cache_dataset)
        category_file = os.path.join(cache_dir, 'category-%s.json' % cache_dataset)
        vocabulary_file = os.path.join(cache_dir, 'vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + cache_dataset + '.json')
        word_embedding_file = os.path.join(cache_dir, 'word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + cache_dataset + '.pkl')
        unit_history_graph_file = os.path.join(cache_dir, 'unit_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + cache_dataset + '.pkl')

        unit_size_file = os.path.join(cache_dir, 'unit_size-%s.json' % cache_dataset)
        unit_type_file = os.path.join(cache_dir, 'unit_type-%s.json' % cache_dataset)
        combat_power_file = os.path.join(cache_dir, 'combat_power-%s.json' % cache_dataset)
        location_file = os.path.join(cache_dir, 'location-%s.json' % cache_dataset)

        preprocessed_data_files = [unit_ID_file, report_ID_file, category_file, vocabulary_file, word_embedding_file, unit_history_graph_file, unit_size_file, unit_type_file, combat_power_file, location_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            unit_ID_dict = {'<UNK>': 0}
            report_ID_dict = {'<PAD>': 0}
            category_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
            word_counter = collections.Counter()
            report_category_dict = {}

            unit_size_dict = {'<UNK>': 0}
            unit_type_dict = {'<UNK>': 0}
            combat_power_dict = {'<UNK>': 0}
            location_dict = {'<UNK>': 0}

            # 1. unit ID dictionay
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                units = Command_unitCorpus._load_units(os.path.join(prefix, 'unit.tsv'))

                for unit in units:
                    unit_id = unit.get("unitId", None)
                    if unit_id is None:
                        continue

                    unit_id = str(unit_id).strip()
                    if unit_id == "":
                        continue

                    if unit_id not in unit_ID_dict:
                        unit_ID_dict[unit_id] = len(unit_ID_dict)

                    unit_size = str(unit.get("unitSize", "<UNK>")).strip() or "<UNK>"
                    unit_type = str(unit.get("unitType", "<UNK>")).strip() or "<UNK>"
                    combat_power = str(unit.get("combatPower", "<UNK>")).strip() or "<UNK>"
                    location = str(unit.get("location", "<UNK>")).strip() or "<UNK>"

                    if unit_size not in unit_size_dict:
                        unit_size_dict[unit_size] = len(unit_size_dict)

                    if unit_type not in unit_type_dict:
                        unit_type_dict[unit_type] = len(unit_type_dict)

                    if combat_power not in combat_power_dict:
                        combat_power_dict[combat_power] = len(combat_power_dict)

                    if location not in location_dict:
                        location_dict[location] = len(location_dict)

            with open(unit_ID_file, 'w', encoding='utf-8') as f:
                json.dump(unit_ID_dict, f, ensure_ascii=False)

            with open(unit_size_file, 'w', encoding='utf-8') as f:
                json.dump(unit_size_dict, f, ensure_ascii=False)

            with open(unit_type_file, 'w', encoding='utf-8') as f:
                json.dump(unit_type_dict, f, ensure_ascii=False)

            with open(combat_power_file, 'w', encoding='utf-8') as f:
                json.dump(combat_power_dict, f, ensure_ascii=False)

            with open(location_file, 'w', encoding='utf-8') as f:
                json.dump(location_dict, f, ensure_ascii=False)

            # 2. report ID dictionay & report category dictionay
            all_reports = []
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                reports = Command_unitCorpus._load_commands(os.path.join(prefix, 'commands.tsv'))
                all_reports.extend(reports)
                for report in reports:
                    report_ID = report.get('dataId', None)
                    if report_ID is None:
                        continue
                    report_ID = str(report_ID).strip()
                    if report_ID == "":
                        continue

                    category = report.get('category', 'UNK')
                    category = str(category).strip() if category is not None else 'UNK'
                    if category == "":
                        category = "UNK"

                    title     = report.get('title', '')
                    content   = report.get('body', '')
                    time_str  = report.get('reportTime', '')

                    title = "" if title is None else str(title)
                    content = "" if content is None else str(content)
                    time_str = "" if time_str is None else str(time_str)

                    if report_ID not in report_ID_dict:
                        report_ID_dict[report_ID] = len(report_ID_dict)

                    if category not in category_dict:
                        category_dict[category] = len(category_dict)

                    report_category_dict[report_ID] = category_dict[category]
                    
                    if config.tokenizer != 'SentencePiece':
                        for text in [str(title).lower(), str(content).lower(), str(time_str).lower()]:
                            toks = mecab.morphs(text) if (config.tokenizer == 'MeCab' and mecab) else word_tokenize(text)
                            for w in toks:
                                if is_number(w):
                                    word_counter['<NUM>'] += 1
                                else:
                                    word_counter[w] += 1
                    

            with open(report_ID_file, 'w', encoding='utf-8') as report_ID_f:
                json.dump(report_ID_dict, report_ID_f, ensure_ascii=False)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f, ensure_ascii=False)

            # 3. word dictionay
            if config.tokenizer != 'SentencePiece':
                word_counter_list = [[word, word_counter[word]] for word in word_counter]
                word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
                filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))

                start_index = len(word_dict)
                add_i = 0

                for w, _cnt in filtered_word_counter_list:
                    if w in word_dict:
                        continue
                    word_dict[w] = start_index + add_i
                    add_i += 1
                with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                    json.dump(word_dict, vocabulary_f, ensure_ascii=False)

           
            if config.tokenizer == 'SentencePiece':
                if spm is None:
                    print('Warning: sentencepiece is not installed. Install it via `pip install sentencepiece` to use SentencePiece tokenizer.')
                else:
                    #corpus_path = f'spm_corpus_{cache_dataset}.txt'
                    corpus_path = os.path.join(cache_dir, f'spm_corpus_{cache_dataset}.txt')
                    with open(corpus_path, 'w', encoding='utf-8') as corpus_f:
                        for prefix in [config.train_root, config.dev_root, config.test_root]:
                            reports = Command_unitCorpus._load_commands(os.path.join(prefix, 'commands.tsv'))
                            for report in reports:
                                t = report.get('title', '') or ''
                                b = report.get('body', '') or ''
                                ti = report.get('reportTime', '') or ''
                                corpus_f.write(str(t) + '\n')
                                corpus_f.write(str(b) + '\n')
                                corpus_f.write(str(ti) + '\n')

                    #model_prefix = f'spm_{cache_dataset}'
                    model_prefix = os.path.join(cache_dir, f'spm_{cache_dataset}')
                    # choose a reasonable requested vocab size
                    requested_vocab = 80000
                    # compute unique whitespace tokens in corpus as an upper bound
                    unique_tokens = set()
                    try:
                        with open(corpus_path, 'r', encoding='utf-8') as cf:
                            for line in cf:
                                for tok in line.strip().split():
                                    if tok:
                                        unique_tokens.add(tok)
                    except Exception:
                        unique_tokens = set()

                    max_vocab_from_corpus = max(100, len(unique_tokens))
                    # cap vocab to safe upper bound to avoid SentencePiece internal limits
                    safe_cap = 80000
                    final_vocab = min(requested_vocab, max_vocab_from_corpus, safe_cap)
                    if final_vocab < requested_vocab:
                        print(f'Adjusting SentencePiece vocab_size from {requested_vocab} to {final_vocab} based on corpus tokens ({len(unique_tokens)} unique tokens) and safe cap {safe_cap}.')
                    spm.SentencePieceTrainer.Train(
                        f"--input={corpus_path} "
                        f"--model_prefix={model_prefix} "
                        f"--vocab_size={final_vocab} "
                        f"--model_type=unigram "
                        f"--character_coverage=0.9995 "
                        f"--hard_vocab_limit=false"
                    )
                    print(f'SentencePiece model trained: {model_prefix}.model ({final_vocab} vocab)')

            # 4. Embedding 생성: MeCab/SentencePiece는 FastText, 나머지는 GloVe
            if config.tokenizer in ['MeCab', 'SentencePiece']:
                
                print(f"{config.tokenizer} 기반 FastText 임베딩 학습 중...")

                # 문장(piece/형태소) 시퀀스 만들기
                sentences = []

                if config.tokenizer == 'MeCab':
                    if mecab is None:
                        raise RuntimeError("MeCab tokenizer를 선택했지만 mecab이 로드되지 않았습니다.")
                    for prefix in [config.train_root, config.dev_root, config.test_root]:
                        reports = Command_unitCorpus._load_commands(os.path.join(prefix, 'commands.tsv'))
                        for r in reports:
                            t = (r.get('title', '') or '').lower()
                            b = (r.get('body', '') or '').lower()
                            ti = (r.get('reportTime', '') or '').lower()
                            text = f"{t} {b} {ti}".strip()
                            if not text:
                                continue
                            tokens = mecab.morphs(text)
                            tokens = [('<NUM>' if is_number(x) else x) for x in tokens]
                            sentences.append(tokens)

                    ft_min_count = config.word_threshold  # MeCab은 보통 threshold 유지 가능

                else:  # SentencePiece
                    if spm is None:
                        raise RuntimeError("SentencePiece tokenizer를 선택했지만 sentencepiece가 설치되지 않았습니다.")
                
                    #model_prefix = f'spm_{cache_dataset}'
                    model_prefix = os.path.join(cache_dir, f'spm_{cache_dataset}')
                    sp_model_file = model_prefix + '.model'
                    sp_proc = spm.SentencePieceProcessor()
                    sp_proc.Load(sp_model_file)

                    # SentencePiece면 vocab(word_dict)을 piece vocab으로 재구성해야 함 (필수)
                    new_word_dict = {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
                    piece_size = sp_proc.get_piece_size() if hasattr(sp_proc, "get_piece_size") else sp_proc.GetPieceSize()

                    for i in range(piece_size):
                        p = sp_proc.id_to_piece(i) if hasattr(sp_proc, "id_to_piece") else sp_proc.IdToPiece(i)
                        if p not in new_word_dict:
                            new_word_dict[p] = len(new_word_dict)

                    word_dict = new_word_dict  # 기존 word_dict(공백/word_tokenize 기반)를 덮어씀

                    # vocab 파일도 덮어써서 __init__이 같은 vocab을 읽게 함
                    with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                        json.dump(word_dict, vocabulary_f, ensure_ascii=False)
                    for fp in [word_embedding_file]:
                        if os.path.exists(fp):
                            try:
                                os.remove(fp)
                            except Exception:
                                pass

                    for prefix in [config.train_root, config.dev_root, config.test_root]:
                        reports = Command_unitCorpus._load_commands(os.path.join(prefix, 'commands.tsv'))
                        for r in reports:
                            t = (r.get('title', '') or '').lower()
                            b = (r.get('body', '') or '').lower()
                            ti = (r.get('reportTime', '') or '').lower()
                            text = f"{t} {b} {ti}".strip()
                            if not text:
                                continue
                            pieces = sp_proc.encode(text, out_type=str)
                            pieces = [('<NUM>' if is_number(x) else x) for x in pieces]
                            sentences.append(pieces)

                    ft_min_count = 1  # SentencePiece는 희귀 piece가 많아서 1 추천

                sentences.append(['<NUM>'])
                # FastText 학습
                ft_model = FastText(
                    sentences=sentences,
                    vector_size=config.word_embedding_dim,
                    window=5,
                    min_count=ft_min_count,
                    workers=4,
                    sg=1
                )

                # 임베딩 매트릭스 생성
                embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
                for token, idx in word_dict.items():
                    if idx == 0:
                        continue
                    if token in ft_model.wv:
                        embedding_vectors[idx] = torch.tensor(ft_model.wv.get_vector(token))
                    else:
                        rv = torch.zeros(config.word_embedding_dim)
                        rv.normal_(mean=0, std=0.1)
                        embedding_vectors[idx] = rv

                with open(word_embedding_file, 'wb') as f:
                    pickle.dump(embedding_vectors, f)
                             
            else:
                print(">>> Using GloVe embedding")
                # Default behavior: word-based vocabulary -> GloVe mapping
                if config.word_embedding_dim == 300:
                    glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
                else:
                    glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='../glove', max_vectors=10000000000)
                glove_stoi = glove.stoi
                glove_vectors = glove.vectors
                glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
                word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
                for word in word_dict:
                    index = word_dict[word]
                    if index != 0:
                        if word in glove_stoi:
                            word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
                        else:
                            random_vector = torch.zeros(config.word_embedding_dim)
                            random_vector.normal_(mean=0, std=0.1)
                            word_embedding_vectors[index, :] = random_vector + glove_mean_vector
                with open(word_embedding_file, 'wb') as word_embedding_f:
                    pickle.dump(word_embedding_vectors, word_embedding_f)

            # 5. unit history graph
            category_num = len(category_dict)
            graph_size = config.max_history_num + category_num # graph size of |V_{n}|+|V_{p}|
            prefix_mode = ['train', 'dev', 'test']
            unit_history_graph_data = {}
            unit_history_orders = {}
            for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                mode = prefix_mode[prefix_index]

                units = Command_unitCorpus._load_units(os.path.join(prefix, 'unit.tsv'))
                unit_history_items = []
                for unit in units:
                    uid = str(unit.get("unitId", "")).strip()
                    history_list = Command_unitCorpus._history_from_str(unit.get("history", ""))
                    if uid:
                        unit_history_items.append((uid, history_list))
                unit_history_orders[mode] = [uid for uid, _ in unit_history_items]
                unit_history_num = len(unit_history_items)

                unit_history_graph = np.zeros([unit_history_num, graph_size, graph_size], dtype=np.float32)
                unit_history_category_mask = np.zeros([unit_history_num, category_num + 1], dtype=bool)
                unit_history_category_indices = np.zeros([unit_history_num, config.max_history_num], dtype=np.int64)
                
                for line_index, (unit_id, history_list) in enumerate(unit_history_items):

                    if not isinstance(history_list, list):
                        continue

                    if config.no_self_connection:
                        history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                    else:
                        history_graph = np.identity(graph_size, dtype=np.float32)
                    history_category_mask = np.zeros(category_num + 1, dtype=bool) # extra one category index for padding news
                    history_category_indices = np.full([config.max_history_num], category_num, dtype=np.int64)
                    if history_list and len(history_list) > 0:
                        history_report_ID = history_list
                        offset = max(0, len(history_report_ID) - config.max_history_num)
                        history_report_num = min(len(history_report_ID), config.max_history_num)
                        for i in range(history_report_num):
                            rid = history_report_ID[i + offset]
                            rid = str(rid).strip() if rid is not None else ""
                            if rid == "" or rid not in report_category_dict:
                                continue
                            category_index = report_category_dict[rid]
                            history_category_mask[category_index] = 1
                            history_category_indices[i] = category_index
                            history_graph[i, config.max_history_num + category_index] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                            history_graph[config.max_history_num + category_index, i] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                            for j in range(i + 1, history_report_num):
                                rid2 = history_report_ID[j + offset]
                                rid2 = str(rid2).strip() if rid2 is not None else ""
                                if rid2 == "" or rid2 not in report_category_dict:
                                    continue
                                _category_index = report_category_dict[rid2]
                                if category_index == _category_index:
                                    history_graph[i, j] = 1 # edge of E_{n} in intra-cluster graph G1
                                    history_graph[j, i] = 1 # edge of E_{n} in intra-cluster graph G1
                                else:
                                    history_graph[config.max_history_num + category_index, config.max_history_num + _category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                                    history_graph[config.max_history_num + _category_index, config.max_history_num + category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                        if not config.no_adjacent_normalization:
                            deg = history_graph.sum(axis=1, keepdims=False)
                            deg[deg == 0] = 1
                            if config.gcn_normalization_type == 'asymmetric':
                                # Asymmetric adjacent matrix normalization: D^{-1}A
                                D_inv = np.zeros([graph_size, graph_size], dtype=np.float32)
                                np.fill_diagonal(D_inv, 1 / deg)
                                history_graph = np.matmul(D_inv, history_graph)
                            else:
                                # Symmetric adjacent matrix normalization: D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
                                D_inv_sqrt = np.zeros([graph_size, graph_size], dtype=np.float32)
                                np.fill_diagonal(D_inv_sqrt, np.sqrt(1 / deg))
                                history_graph = np.matmul(np.matmul(D_inv_sqrt, history_graph), D_inv_sqrt)
                    unit_history_graph[line_index] = history_graph
                    unit_history_category_mask[line_index] = history_category_mask
                    unit_history_category_indices[line_index] = history_category_indices
                unit_history_graph_data[mode + '_unit_history_graph'] = unit_history_graph
                unit_history_graph_data[mode + '_unit_history_category_mask'] = unit_history_category_mask
                unit_history_graph_data[mode + '_unit_history_category_indices'] = unit_history_category_indices
            unit_history_graph_data['unit_history_orders'] = unit_history_orders
            with open(unit_history_graph_file, 'wb') as unit_history_graph_f:
                pickle.dump(unit_history_graph_data, unit_history_graph_f)

    def __init__(self, config: Config):
        # preprocess data
        Command_unitCorpus.preprocess(config)
        # 26.05 추가
        cache_dataset = config.dataset

        if getattr(config, "time_eval", False):
            cache_dir = os.path.join("cache", "time", config.dataset, "all")
            cache_dataset = f"{config.dataset}-time"
        else:
            cache_dir = os.path.join("cache", "unit", config.dataset)
            cache_dataset = f"{config.dataset}-unit"

        os.makedirs(cache_dir, exist_ok=True)

        with open(os.path.join(cache_dir, 'unit_ID-%s.json' % cache_dataset), 'r', encoding='utf-8') as f:
            self.unit_ID_dict = json.load(f)
            self.unit_num = len(self.unit_ID_dict)
        with open(os.path.join(cache_dir, 'report_ID-%s.json' % cache_dataset), 'r', encoding='utf-8') as report_ID_f:
            self.report_ID_dict = json.load(report_ID_f)
            self.report_num = len(self.report_ID_dict)
        with open(os.path.join(cache_dir, 'category-%s.json' % cache_dataset), 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open(os.path.join(cache_dir, 'vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + cache_dataset + '.json'), 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)

        with open(os.path.join(cache_dir, 'unit_size-%s.json' % cache_dataset), 'r', encoding='utf-8') as f:
            self.unit_size_dict = json.load(f)
            config.unit_size_num = len(self.unit_size_dict)

        with open(os.path.join(cache_dir, 'unit_type-%s.json' % cache_dataset), 'r', encoding='utf-8') as f:
            self.unit_type_dict = json.load(f)
            config.unit_type_num = len(self.unit_type_dict)

        with open(os.path.join(cache_dir, 'combat_power-%s.json' % cache_dataset), 'r', encoding='utf-8') as f:
            self.combat_power_dict = json.load(f)
            config.combat_power_num = len(self.combat_power_dict)

        with open(os.path.join(cache_dir, 'location-%s.json' % cache_dataset), 'r', encoding='utf-8') as f:
            self.location_dict = json.load(f)
            config.location_num = len(self.location_dict)
        
        with open(os.path.join(cache_dir, 'unit_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + cache_dataset + '.pkl'), 'rb') as unit_history_graph_f:
            unit_history_data = pickle.load(unit_history_graph_f)
            self.train_unit_history_graph = unit_history_data['train_unit_history_graph']
            self.train_unit_history_category_mask = unit_history_data['train_unit_history_category_mask']
            self.train_unit_history_category_indices = unit_history_data['train_unit_history_category_indices']
            self.dev_unit_history_graph = unit_history_data['dev_unit_history_graph']
            self.dev_unit_history_category_mask = unit_history_data['dev_unit_history_category_mask']
            self.dev_unit_history_category_indices = unit_history_data['dev_unit_history_category_indices']
            self.test_unit_history_graph = unit_history_data['test_unit_history_graph']
            self.test_unit_history_category_mask = unit_history_data['test_unit_history_category_mask']
            self.test_unit_history_category_indices = unit_history_data['test_unit_history_category_indices']
            self.unit_history_orders = unit_history_data['unit_history_orders']


        # meta data
        self.negative_sample_num = config.negative_sample_num                                           # negative sample number for training
        self.max_history_num = config.max_history_num                                                   # max history number for each training unit
        self.max_title_length = config.max_title_length                                                 # max title length for each news text
        self.max_content_length = config.max_content_length                                             # max content length for each news text
        self.max_time_length = config.max_time_length                                                   # max time length for each news text      
        
        self.report_category = np.zeros([self.report_num], dtype=np.int32)                                  # [report_num]
        self.report_title_text = np.zeros([self.report_num, self.max_title_length], dtype=np.int32)         # [report_num, max_title_length]
        self.report_title_mask = np.zeros([self.report_num, self.max_title_length], dtype=bool)             # [report_num, max_title_length]
        self.report_content_text = np.zeros([self.report_num, self.max_content_length], dtype=np.int32)   # [report_num, max_content_length]
        self.report_content_mask = np.zeros([self.report_num, self.max_content_length], dtype=bool)       # [report_num, max_content_length]
        self.report_time_text = np.zeros([self.report_num, self.max_time_length], dtype=np.int32)
        self.report_time_mask = np.zeros([self.report_num, self.max_time_length], dtype=bool)
        self.report_valid_until = np.zeros([self.report_num], dtype=np.int32)
        self.report_security_level = np.zeros([self.report_num], dtype=np.int32)
        self.report_time_ts = np.zeros([self.report_num], dtype=np.int64)


        self.train_unitDataset = []                                                                       # [unit_ID, [history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_unitDataset = []                                                                         # [unit_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.dev_indices = []                                                                            # index for dev
        self.test_unitDataset = []                                                                        # [unit_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.test_indices = []

        self.unit_size = np.zeros([self.unit_num], dtype=np.int32)
        self.unit_type = np.zeros([self.unit_num], dtype=np.int32)
        self.unit_combat_power = np.zeros([self.unit_num], dtype=np.int32)
        self.unit_location = np.zeros([self.unit_num], dtype=np.int32)
        
        for prefix in [config.train_root, config.dev_root, config.test_root]:
            for unit in Command_unitCorpus._load_units(os.path.join(prefix, 'unit.tsv')):
                unit_id = unit.get("unitId", None)
                if unit_id is None:
                    continue
                unit_id = str(unit_id).strip()
                if unit_id == "" or unit_id not in self.unit_ID_dict:
                    continue

                uidx = self.unit_ID_dict[unit_id]

                unit_size = str(unit.get("unitSize", "<UNK>")).strip() or "<UNK>"
                unit_type = str(unit.get("unitType", "<UNK>")).strip() or "<UNK>"
                combat_power = str(unit.get("combatPower", "<UNK>")).strip() or "<UNK>"
                location = str(unit.get("location", "<UNK>")).strip() or "<UNK>"

                self.unit_size[uidx] = self.unit_size_dict.get(unit_size, 0)
                self.unit_type[uidx] = self.unit_type_dict.get(unit_type, 0)
                self.unit_combat_power[uidx] = self.combat_power_dict.get(combat_power, 0)
                self.unit_location[uidx] = self.location_dict.get(location, 0)

        self.title_word_num = 0
        self.content_word_num = 0
        self.time_word_num = 0

        # generate news meta data
        report_ID_set  = set(['<PAD>'])
        report_items = []
        for prefix in [config.train_root, config.dev_root, config.test_root]:
            reports = Command_unitCorpus._load_commands(os.path.join(prefix, 'commands.tsv'))
            for report in reports:
                report_ID = report.get('dataId', None)
                if report_ID is None:
                    continue
                report_ID = str(report_ID).strip()
                if report_ID == "":
                    continue

                if report_ID not in report_ID_set:
                    report_items.append(report)
                    report_ID_set.add(report_ID)

        assert self.report_num == len(report_ID_set), 'report num mismatch %d v.s. %d' % (self.report_num, len(report_ID_set))

        UNK_IDX = self.word_dict.get('<UNK>', 1)
        NUM_IDX = self.word_dict.get('<NUM>', 2)

        # Prepare SentencePiece processor if available and requested
        sp_proc = None
        if config.tokenizer == 'SentencePiece' and spm is not None:
            #sp_model_file = f'spm_{cache_dataset}.model'
            sp_model_file = os.path.join(cache_dir, f'spm_{cache_dataset}.model')
            if os.path.exists(sp_model_file):
                try:
                    sp_proc = spm.SentencePieceProcessor()
                    sp_proc.Load(sp_model_file)
                except Exception:
                    sp_proc = None

        for report in report_items:
            report_ID = report.get('dataId', None)
            if report_ID is None:
                continue
            report_ID = str(report_ID).strip()
            if report_ID == "":
                continue
            if report_ID not in self.report_ID_dict:
                continue

            category = report.get('category', 'UNK')
            category = str(category).strip() if category is not None else 'UNK'
            if category == "":
                category = 'UNK'

            title = report.get('title', '')
            content = report.get('body', '')
            time_str = report.get('reportTime', '')

            title = "" if title is None else str(title)
            content = "" if content is None else str(content)
            time_str = "" if time_str is None else str(time_str)
            valid_until = report.get('validUntil', None)
            security_lv = report.get('securityLevel', None)

            index = self.report_ID_dict[report_ID]
            self.report_category[index] = self.category_dict.get(category, 0)
            self.report_time_ts[index] = Command_unitCorpus._parse_time_to_ts(time_str)
            
            if valid_until is not None:
                try:
                    self.report_valid_until[index] = int(valid_until)
                except Exception:
                    pass
            if security_lv is not None:
                try:
                    self.report_security_level[index] = int(security_lv)
                except Exception:
                    pass
            # title
            if sp_proc is not None:
                words = sp_proc.encode(title.lower(), out_type=str)
            elif config.tokenizer == 'MeCab' and mecab:
                words = mecab.morphs(title.lower())
            else:
                words = word_tokenize(title.lower())
            for j, word in enumerate(words):
                if j >= self.max_title_length:
                    break
                if is_number(word):
                    self.report_title_text[index][j] = NUM_IDX
                else:
                    self.report_title_text[index][j] = self.word_dict.get(word, UNK_IDX)
                self.report_title_mask[index][j] = True
            self.title_word_num += len(words)
            # content
            if sp_proc is not None:
                words = sp_proc.encode(content.lower(), out_type=str)
            elif config.tokenizer == 'MeCab' and mecab:
                words = mecab.morphs(content.lower())
            else:
                words = word_tokenize(content.lower())
            for j, word in enumerate(words):
                if j >= self.max_content_length:
                    break
                if is_number(word):
                    self.report_content_text[index][j] = NUM_IDX
                else:
                    self.report_content_text[index][j] = self.word_dict.get(word, UNK_IDX)
                self.report_content_mask[index][j] = True
            self.content_word_num += len(words)
            # time
            if sp_proc is not None:
                words = sp_proc.encode(time_str.lower(), out_type=str)
            elif config.tokenizer == 'MeCab' and mecab:
                words = mecab.morphs(time_str.lower())
            else:
                words = word_tokenize(time_str.lower())
            for j, word in enumerate(words):
                if j >= self.max_time_length:
                    break
                if is_number(word):
                    self.report_time_text[index][j] = NUM_IDX
                else:
                    self.report_time_text[index][j] = self.word_dict.get(word, UNK_IDX)
                self.report_time_mask[index][j] = True
            self.time_word_num += len(words)

        self.report_title_mask[0][0] = True    # for <PAD> report
        self.report_content_mask[0][0] = True  # for <PAD> report
        self.report_time_mask[0][0] = True     # for <PAD> report

        def _make_unit_history_tables(root_dir: str):
            hist_index = np.zeros([self.unit_num, self.max_history_num], dtype=np.int64)
            hist_mask  = np.zeros([self.unit_num, self.max_history_num], dtype=bool)

            units = Command_unitCorpus._load_units(os.path.join(root_dir, "unit.tsv"))
            for u in units:
                uid = str(u.get("unitId", "")).strip()
                if not uid:
                    continue
                uidx = self.unit_ID_dict.get(uid, 0)
                if uidx == 0:
                    continue

                history_list = Command_unitCorpus._history_from_str(u.get("history", ""))
                if not history_list:
                    continue

                # keep last max_history_num (MIND 스타일)
                offset = max(0, len(history_list) - self.max_history_num)
                history_list = history_list[offset:]

                for j, rid in enumerate(history_list):
                    if j >= self.max_history_num:
                        break
                    rid = "" if rid is None else str(rid).strip()
                    if not rid:
                        continue
                    r_idx = self.report_ID_dict.get(rid, 0)  # unknown -> 0(PAD)
                    hist_index[uidx, j] = r_idx
                    if r_idx != 0:
                        hist_mask[uidx, j] = True

            return hist_index, hist_mask

        def _build_unit_history_sequences(root_dir: str):
            # uidx -> (hist_idx_list_sorted_by_time, hist_ts_list_sorted_by_time)
            seq_idx = [[] for _ in range(self.unit_num)]
            seq_ts  = [[] for _ in range(self.unit_num)]

            units = Command_unitCorpus._load_units(os.path.join(root_dir, "unit.tsv"))
            for u in units:
                uid = str(u.get("unitId", "")).strip()
                if not uid:
                    continue
                uidx = self.unit_ID_dict.get(uid, 0)
                if uidx == 0:
                    continue

                history_list = Command_unitCorpus._history_from_str(u.get("history", ""))
                for rid in history_list:
                    rid = "" if rid is None else str(rid).strip()
                    if not rid:
                        continue
                    r_idx = self.report_ID_dict.get(rid, 0)
                    if r_idx == 0:
                        continue
                    ts = int(self.report_time_ts[r_idx])
                    if ts <= 0:
                        continue
                    seq_idx[uidx].append(r_idx)
                    seq_ts[uidx].append(ts)

                # 시간순 정렬(bisect를 위해)
                if len(seq_ts[uidx]) > 1:
                    order = np.argsort(np.asarray(seq_ts[uidx], dtype=np.int64), kind="mergesort")
                    seq_idx[uidx] = [seq_idx[uidx][k] for k in order]
                    seq_ts[uidx]  = [seq_ts[uidx][k]  for k in order]

            return seq_idx, seq_ts

        # split별 unit history 시퀀스(시간순)
        self.train_unit_hist_seq_idx, self.train_unit_hist_seq_ts = _build_unit_history_sequences(config.train_root)
        self.dev_unit_hist_seq_idx,   self.dev_unit_hist_seq_ts   = _build_unit_history_sequences(config.dev_root)
        self.test_unit_hist_seq_idx,  self.test_unit_hist_seq_ts  = _build_unit_history_sequences(config.test_root)

        def _get_history_before(mode: str, uidx: int, cutoff_ts: int):
            if uidx <= 0 or cutoff_ts <= 0:
                return np.zeros([self.max_history_num], dtype=np.int64), np.zeros([self.max_history_num], dtype=bool)

            if mode == 'train':
                seq_idx, seq_ts = self.train_unit_hist_seq_idx, self.train_unit_hist_seq_ts
            elif mode == 'dev':
                seq_idx, seq_ts = self.dev_unit_hist_seq_idx, self.dev_unit_hist_seq_ts
            else:
                seq_idx, seq_ts = self.test_unit_hist_seq_idx, self.test_unit_hist_seq_ts

            ts_list = seq_ts[uidx]
            idx_list = seq_idx[uidx]
            if not ts_list:
                return np.zeros([self.max_history_num], dtype=np.int64), np.zeros([self.max_history_num], dtype=bool)

            k = bisect.bisect_left(ts_list, cutoff_ts)  # cutoff 이전(<)만
            if k <= 0:
                return np.zeros([self.max_history_num], dtype=np.int64), np.zeros([self.max_history_num], dtype=bool)

            valid = idx_list[:k]
            valid = valid[-self.max_history_num:]  # 마지막 max_history_num개

            out_idx = np.zeros([self.max_history_num], dtype=np.int64)
            out_msk = np.zeros([self.max_history_num], dtype=bool)
            out_idx[:len(valid)] = np.asarray(valid, dtype=np.int64)
            out_msk[:len(valid)] = True
            return out_idx, out_msk

        self.get_history_before = _get_history_before


        self.train_unit_history_index, self.train_unit_history_mask = _make_unit_history_tables(config.train_root)
        self.dev_unit_history_index,  self.dev_unit_history_mask  = _make_unit_history_tables(config.dev_root)
        self.test_unit_history_index, self.test_unit_history_mask = _make_unit_history_tables(config.test_root)

        self.train_unitDataset = []  # (cmd_idx, pos_unit, neg_pool, behavior_index)
        self.train_behaviors_time_ts = {}

        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            for behavior_index, line in enumerate(f):
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue

                impression_ID, dataId, time, impression_units = parts[0], parts[1], parts[2], parts[3]

                cutoff_ts = Command_unitCorpus._parse_time_to_ts(time)
                self.train_behaviors_time_ts[behavior_index] = cutoff_ts

                cmd_idx = self.report_ID_dict.get(dataId, 0)
                if cmd_idx == 0:
                    continue

                pos_units = []
                neg_pool = []

                for token in impression_units.split():
                    if '-' not in token:
                        continue

                    unit_id, label = token.rsplit('-', 1)
                    unit_idx = self.unit_ID_dict.get(unit_id, 0)

                    if unit_idx == 0:
                        continue

                    if label == '1':
                        pos_units.append(unit_idx)
                    else:
                        neg_pool.append(unit_idx)

                for pos_unit in pos_units:
                    self.train_unitDataset.append((cmd_idx, pos_unit, neg_pool, behavior_index))


        # ---- dev ----
        self.dev_unitDataset = []
        self.dev_indices = []
        self.dev_labels = {}
        self.dev_behaviors_time_ts = {}

        with open(os.path.join(config.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            for dev_ID, line in enumerate(f):
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue

                impression_ID, dataId, time, impression_units = parts[0], parts[1], parts[2], parts[3]

                cutoff_ts = Command_unitCorpus._parse_time_to_ts(time)
                self.dev_behaviors_time_ts[dev_ID] = cutoff_ts

                cmd_idx = self.report_ID_dict.get(dataId, 0)
                if cmd_idx == 0:
                    continue

                cand_units = []
                labels = []

                for token in impression_units.split():
                    if '-' not in token:
                        continue

                    unit_id, label = token.rsplit('-', 1)
                    unit_idx = self.unit_ID_dict.get(unit_id, 0)

                    if unit_idx == 0:
                        continue

                    cand_units.append(unit_idx)
                    labels.append(1 if label == '1' else 0)

                if len(cand_units) == 0:
                    continue

                self.dev_unitDataset.append([cmd_idx, cand_units, dev_ID])
                self.dev_indices.append(dev_ID)
                self.dev_labels[dev_ID] = labels

        # ---- test ----
        self.test_unitDataset = []  
        self.test_indices = []
        self.test_labels = {}        
        self.test_behaviors_time_ts = {}

        with open(os.path.join(config.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            for test_ID, line in enumerate(f):
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue

                impression_ID, dataId, time, impression_units = parts[0], parts[1], parts[2], parts[3]

                cutoff_ts = Command_unitCorpus._parse_time_to_ts(time)
                self.test_behaviors_time_ts[test_ID] = cutoff_ts

                cmd_idx = self.report_ID_dict.get(dataId, 0)
                if cmd_idx == 0:
                    continue

                cand_units = []
                labels = []

                for token in impression_units.split():
                    if '-' not in token:
                        continue

                    unit_id, label = token.rsplit('-', 1)
                    unit_idx = self.unit_ID_dict.get(unit_id, 0)

                    if unit_idx == 0:
                        continue

                    cand_units.append(unit_idx)
                    labels.append(1 if label == '1' else 0)

                if len(cand_units) == 0:
                    continue

                self.test_unitDataset.append([cmd_idx, cand_units, test_ID])
                self.test_indices.append(test_ID)
                self.test_labels[test_ID] = labels

        self.train_unitidx_to_graphrow = Command_unitCorpus._build_unitidx_to_graphrow(
            self.unit_history_orders.get('train', []), self.unit_ID_dict
        )

        self.dev_unitidx_to_graphrow = Command_unitCorpus._build_unitidx_to_graphrow(
            self.unit_history_orders.get('dev', []), self.unit_ID_dict
        )
        self.test_unitidx_to_graphrow = Command_unitCorpus._build_unitidx_to_graphrow(
            self.unit_history_orders.get('test', []), self.unit_ID_dict
        )