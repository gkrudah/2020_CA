import torch
from tqdm import tqdm
from config import device


def eval(net, dataloader):
    right = 0
    total = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            out = net(x)
            out_labels = out.argmax(dim=1)
            rights = (out_labels == y).float()
            right += rights.sum().item()
            total += out.size(0)

    return right / total * 100