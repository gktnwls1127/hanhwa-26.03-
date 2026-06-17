import time
import os
import gc
import shutil
from config import Config
import torch
from Command_corpus import Command_Corpus
from model import Model
from trainer import Trainer, distributed_train
from util import compute_scores, get_run_index
import torch.multiprocessing as mp


def train(config: Config, command_corpus: Command_Corpus):
    model = Model(config)
    model.initialize()
    run_index = get_run_index(config.result_dir)
    if config.world_size == 1:
        trainer = Trainer(model, config, command_corpus, run_index)
        trainer.train()
        trainer = None
        del trainer
    else:
        try:
            mp.spawn(distributed_train, args=(model, config, command_corpus, run_index), nprocs=config.world_size, join=True)
        except Exception as e:
            print(e)
            e = str(e).lower()
            if 'cuda' in e or 'pytorch' in e:
                exit()
    config.run_index = run_index
    model = None
    del model
    gc.collect()
    if config.gpu_available:
        torch.cuda.empty_cache()


def dev(config: Config, command_corpus: Command_Corpus):
    model = Model(config)
    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    if config.gpu_available:
        model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, command_corpus, config, config.batch_size * 2 // config.world_size, 'dev', dev_res_dir + '/' + model.model_name + '.txt')
    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10


def test(config: Config, command_corpus: Command_Corpus):
    if config.gpu_available:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    model = Model(config)
    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    if config.gpu_available:
        model.cuda()
    test_res_dir = os.path.join(config.test_res_dir, config.test_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)
    print('test model path  : ' + config.test_model_path)
    print('test output file : ' + test_res_dir + '/' + model.model_name + '.txt')

    if config.gpu_available:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    if config.report_encoder == 'CROWN' and config.content_encoder == 'CROWN':
        test_batch_size = 1
    else:
        test_batch_size = config.batch_size * 2 // config.world_size
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, command_corpus, config, test_batch_size, 'test', test_res_dir + '/' + model.model_name + '.txt')
    
    #auc, mrr, ndcg5, ndcg10 = compute_scores(model, command_corpus, config, config.batch_size * 2 // config.world_size, 'test', test_res_dir + '/' + model.model_name + '.txt')
    
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    if config.mode == 'train':
        with open(config.result_dir + '/#' + str(config.run_index) + '-test', 'w') as result_f:
            result_f.write('#' + str(config.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    elif config.mode == 'test' and config.test_output_file != '':
        with open(config.test_output_file, 'w', encoding='utf-8') as f:
            f.write('#' + str(config.seed + 1) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    if config.dataset == 'large':
        if config.mode == 'train':
            shutil.copy(test_res_dir + '/' + model.model_name + '.txt', 'prediction/large/%s/#%d/prediction.txt' % (model.model_name, config.run_index))
            os.chdir('prediction/large/%s/#%d' % (model.model_name, config.run_index))
            # Use Python's zipfile instead of os.system for cross-platform compatibility
            import zipfile
            with zipfile.ZipFile('prediction.zip', 'w') as zipf:
                zipf.write('prediction.txt')
            os.chdir('../../../..')

    if config.gpu_available:
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    test_time = format_mmss(t2 - t1)
    run_index = getattr(config, 'run_index', config.seed + 1)

    with open(config.result_dir + f"/#{config.run_index}-test_time.txt", "w") as f:
        f.write(f"TEST\t{test_time}\n")

    print(f"[Time] TEST: {test_time}")
    #print(f"[Time] TEST_MAIN_TOTAL: {t2 - t0:.2f}s")   #  로드~끝까지
    #print(f"[Time] TEST_AFTER_PRINT: {t2 - t1:.2f}s")  #  'test model path 출력 이후' 기준

def format_mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


if __name__ == '__main__':
    config = Config()
    if config.mode == 'train' and config.time_eval:
        for idx in range(1, 10):
            print(f"\n========== Temporal Train {idx}/9 ==========")

            config.time_train_idx = idx
            config.train_root = '../Command-%s/time/train_%d' % (config.dataset, idx)
            config.dev_root   = '../Command-%s/time/test' % config.dataset
            config.test_root  = '../Command-%s/time/test' % config.dataset

            command_corpus = Command_Corpus(config)
            train(config, command_corpus)
            config.test_model_path = (config.best_model_dir + '/#' +str(config.run_index) + '/' + config.report_encoder + '-' + config.user_encoder)
            test(config, command_corpus)

            del command_corpus
            gc.collect()
            if config.gpu_available:
                torch.cuda.empty_cache()

    else:
        command_corpus = Command_Corpus(config)
        if config.mode == 'train':
            train(config, command_corpus)
            # LIME 추가(26.05)
            if config.report_encoder == 'LIME':
                model_name = config.report_encoder + '-' + config.content_encoder + '-' + config.user_encoder
            else:
                model_name = config.report_encoder + '-' + config.user_encoder

            config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + model_name
            #config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + config.report_encoder + '-' + config.user_encoder
            test(config, command_corpus)
        elif config.mode == 'dev':
            dev(config, command_corpus)
        elif config.mode == 'test':
            test(config, command_corpus)