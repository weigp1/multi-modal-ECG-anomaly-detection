from easydict import EasyDict as edict

config = edict()
config.gpu = '0'
config.save_frequency = 1
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
config.data.seq_len = 15
config.data.num_worker = 0
config.data.input_type = ''
config.data.test_id = 0

# network
config.net = edict()
config.net.name = 'bilstm'
config.net.hand_craft = False
config.net.alpha = 0.3

config.net.num_hidden = 18
config.net.input_dim = 9
config.net.aux_dim = 4
config.net.hand_dim = 0

# train
config.train = edict()
config.train.resume_epoch = False
config.train.fine_tune = True
config.train.batch_size = 4
config.train.lr = 0.01
config.train.lr_epoch = [10, 20]
config.train.lr_factor = 0.1
config.train.end_epoch = 5
config.train.callback_freq = 50
config.train.optimizer = 'adam'
config.train.warmup_iters = 0
config.train.lr_mult = 0.2

# test
config.test = edict()
config.test.model_name = 'exp_202410171224_bilstm_ep-0010.pth'
config.test.model_path = './model/'
