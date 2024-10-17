import logging
import os
import argparse
import pprint
from datetime import date
import time

import numpy as np
import random
import warnings
import torch
import torch.utils.data

from config import config
from dataloader.iterator_factory import get_dataiter
from lib import create_logger
from lib import metric

import model_prompt
from dataloader import data_loader
from model_prompt import model

from network.bilstm_prompt import BiLSTMModel


# torch.backends.cudnn.enabled = False


if __name__ == "__main__":

    # 创建日志记录器并设置输出路径
    log_file = 'experiment_{}_{}.log'.format("llama2", time.strftime('%Y-%m-%d-%H-%M'))
    model_fixtime = log_file[-20:-4].replace('-','')

    # 记录配置设置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info('Training configuration:{}\n'.format(pprint.pformat(config)))
    vis = True

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu #enable GPU

    if config.seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        # torch.cuda.manual_seed(config.seed) # GPU
        # torch.backends.cudnn.deterministic = True  # GPU
        # torch.backends.cudnn.benchmark = False  # GPU
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    _, data_iter = get_dataiter(config.data.root, config.data.set, config.train.batch_size,
                                config.data.num_worker, 'clip')

    alldata_loader = torch.utils.data.DataLoader(data_iter, batch_size=config.train.batch_size, num_workers=config.data.num_worker,
                                                pin_memory=True, worker_init_fn=data_loader.worker_init_fn)

    # 根据需要获取提示信息
    pmpt = getattr(data_iter, 'pmpt', None)

    sym_net = BiLSTMModel(X_shape=250, HFF_shape=529, prompt_dict=pmpt)
    sym_net.float()

    model_prefix = os.path.join('exp_' + model_fixtime + '_' + config.net.name)

    # criterion=torch.nn.MSELoss().cuda()       # GPU
    net = model(net=sym_net, criterion=torch.nn.MSELoss(), model_prefix=model_prefix, step_callback_freq=config.train.callback_freq,
                save_checkpoint_freq=config.save_frequency, logger = logger) # CPU
            
    # net.net.cuda() # GPU
    # net.net = torch.nn.DataParallel(net.net).cuda() #GPU

    net.net = torch.nn.DataParallel(net.net) #CPU
    net.test_load_checkpoint(load_path=config.test.model_path, model_name=config.test.model_name)

    metrics = metric.MetricList(metric.RMSE(max_rul=config.data.max_rul), metric.RULscore(max_rul=config.data.max_rul),)
    net.data_iter = alldata_loader
    net.dataset = data_iter
    net.metrics = metrics

    # test loop:
    net.dataset.reset()
    net.metrics.reset()
    net.net.eval()
    sum_sample_inst = 0
    sum_sample_elapse = 0.
    sum_update_elapse = 0
    net.callback_kwargs['prefix'] = 'Test'
    batch_start_time = time.time()

    if vis:
        rul = []
        gt = []

    for i_batch, dats in enumerate(net.data_iter):

        net.callback_kwargs['batch'] = i_batch
        update_start_time = time.time()
        # [forward] making next step
        outputs, losses = net.forward(dats)

        # [evaluation] update train metric
        metrics.update([output.data.cpu() for output in outputs], dats[-1].cpu(),
                        [loss.data.cpu() for loss in losses])

        # timing each batch
        sum_sample_elapse += time.time() - batch_start_time
        sum_update_elapse += time.time() - update_start_time
        batch_start_time = time.time()
        sum_sample_inst += dats[0].shape[0]

        if (i_batch % net.step_callback_freq) == 0:
            # retrive eval results and reset metic
            net.callback_kwargs['namevals'] = net.metrics.get_name_value()
            # speed monitor
            net.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
            net.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
            sum_update_elapse = 0
            sum_sample_elapse = 0
            sum_sample_inst = 0
            # callbacks
            net.step_end_callback()
        
        #record RUL results for visulization
        if vis:
            rul.extend(outputs[0].cpu().numpy()[:,0].tolist())
            gt.extend(dats[-1].numpy()[:,0].tolist())
            res = {'rul': rul, 'gt': gt}
    
    # retrive eval results and reset metic
    net.callback_kwargs['namevals'] = net.metrics.get_name_value()
    # speed monitor
    net.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
    net.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
    # callbacks
    net.step_end_callback()

    if vis:
        import pickle
        res_pt = 'res/{:}_{:}_test.pkl'.format(config.data.set, config.net.name)
        with open(res_pt, 'wb') as f:
            pickle.dump(res, f)
        print("Save the test results at {:}".format(res_pt))