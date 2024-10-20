from easydict import EasyDict as edict

config = edict()
config.gpu = '0'
config.save_frequency = 6
config.seed = 0

# distributed training
config.dist_backend = 'nccl'
config.world_size = -1
config.rank = -1
config.dist_url = 'tcp://224.66.41.62:23456'
config.multiprocessing_distributed = False
config.distributed = False
config.task = ''

# dataloader
config.data = edict()
config.data.root = './dataset/'
config.data.set = 'mitdb'
config.data.num_worker = 0
config.data.input_type = ''
config.data.test_id = 0

# network
config.net = edict()
config.net.name = 'bilstm'
config.net.alpha = 0.7
config.net.dropout = 0.2

config.net.raw_input_dim = 250
config.net.raw_hidden_dim = 128
config.net.raw_hidden_dim_2 = 64
config.net.raw_linear_dim = 32

config.net.feat_input_dim = 529
config.net.feat_hidden_dim = 256
config.net.feat_linear_dim = 64

config.net.prompt_dim = 4096
config.net.output_dim = 5

# train
config.train = edict()
config.train.resume_epoch = False
config.train.fine_tune = True
config.train.batch_size = 5
config.train.lr = 0.01
config.train.lr_epoch = [1, 3, 5]
config.train.lr_factor = 0.1
config.train.end_epoch = 5
config.train.callback_freq = 5
config.train.optimizer = 'adam'
config.train.warmup_iters = 0
config.train.lr_mult = 0.2

# test
config.test = edict()
config.test.model_name = 'exp_202410202136_bilstm_ep-0001.pth'  # 需修改
config.test.model_path = './model/'
