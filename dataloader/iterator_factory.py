import torch.utils.data
from dataloader.data_loader import DataIter, worker_init_fn

def get_dataiter(root, set, batch_size, num_worker, mod='normal'):
    assert mod == 'clip', '不支持的mod'

    data_iter = DataIter(root, set)
    data_loader = torch.utils.data.DataLoader(data_iter, batch_size=batch_size, num_workers=num_worker, pin_memory=True, worker_init_fn=worker_init_fn)

    return data_loader, data_iter
