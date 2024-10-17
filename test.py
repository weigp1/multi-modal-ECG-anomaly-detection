import os
import time
import pprint
import logging
import random
import warnings
import numpy as np
import torch.utils.data

from lib import metric
from config import config
from model_prompt import model
from network.bilstm_prompt import BiLSTMModel
from dataloader.iterator_factory import get_dataiter

# torch.backends.cudnn.enabled = False


if __name__ == "__main__":

    # 创建日志记录器并设置输出路径
    log_file = 'experiment_{}_{}.log'.format("llama2", time.strftime('%Y-%m-%d-%H-%M'))
    model_fixtime = log_file[-20:-4].replace('-','')

    # 记录配置设置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info('Training configuration:{}\n'.format(pprint.pformat(config)))

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu #enable GPU

    if config.seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed) # GPU
        torch.backends.cudnn.deterministic = True  # GPU
        torch.backends.cudnn.benchmark = False  # GPU
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    data_loader, data_iter = get_dataiter(config.data.root, config.data.set, config.train.batch_size, config.data.num_worker, 'clip')

    # 根据需要获取提示信息
    pmpt = getattr(data_iter, 'pmpt', None)

    sym_net = BiLSTMModel(X_shape=250, HFF_shape=529, prompt_dict=pmpt)
    sym_net.float()

    model_prefix = os.path.join('exp_' + model_fixtime + '_' + config.net.name)

    criterion=torch.nn.MSELoss().cuda()       # GPU
    net = model(net=sym_net, criterion=torch.nn.MSELoss(), model_prefix=model_prefix, step_callback_freq=config.train.callback_freq,
                save_checkpoint_freq=config.save_frequency, logger = logger) # CPU
            
    net.net.cuda() # GPU
    net.net = torch.nn.DataParallel(net.net).cuda() #GPU

    net.test_load_checkpoint(load_path=config.test.model_path, model_name=config.test.model_name)

    metrics = metric.MetricList(metric.RMSE())
    net.data_iter = data_loader
    net.dataset = data_iter
    net.metrics = metrics

    # test loop:
    net.dataset.reset('test')
    net.metrics.reset()
    net.net.eval()
    sum_sample_inst = 0
    sum_sample_elapse = 0.
    sum_update_elapse = 0
    net.callback_kwargs['prefix'] = 'Test'
    batch_start_time = time.time()
    total_correct = 0

    for i_batch, dats in enumerate(net.data_iter):

        net.callback_kwargs['batch'] = i_batch
        update_start_time = time.time()
        
        outputs, losses = net.forward(dats)
        # 更新度量指标
        preds = [outputs[0][0].argmax(dim=-1).cpu().numpy()]
        
        mse_loss = torch.pow((torch.tensor(preds[0]) - dats[1].cpu()), 2).mean()
        self.metrics.update(preds, dats[1].cpu(), mse_loss, [loss.data.cpu() for loss in losses])
        
        # 计算accuracy
        truth = dats[1].cpu().numpy()
        correct = (preds == truth).sum()
        total_correct += correct

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
    
    # retrive eval results and reset metic
    net.callback_kwargs['namevals'] = net.metrics.get_name_value()
    accuracy = total_correct / net.dataset.end  # 假设net.dataset有长度属性，或者你需要从其他地方获取总样本数
    print(f'Test Accuracy: {accuracy:.4f}')
    # speed monitor
    net.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
    net.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
    # callbacks
    net.step_end_callback()
