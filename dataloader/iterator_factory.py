import torch.utils.data
from dataloader.data_loader import DataIter

def get_dataiter(root, set, batch_size, num_worker, mod='clip'):
    assert mod == 'clip', '不支持的mod'

    data_iter = DataIter(root, set)
    data_loader = torch.utils.data.DataLoader(data_iter, batch_size=batch_size, pin_memory=True)

    return data_loader, data_iter
