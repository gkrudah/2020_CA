import torch
# import torch.nn as nn
from tqdm import tqdm
import numpy as np

from config import device


def train(net, dataloader, optimizer, loss_func):
    avg_loss = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = net(x)

        loss = loss_func(out.squeeze(), y)

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())

    return np.mean(avg_loss)


def mix_train(net, su_dataloader, un_trainloader, optimizer, loss_func, epochnum, su_batch, un_batch, t2=50):
    avg_loss = []
    pbar = tqdm(zip(su_dataloader, un_trainloader), total=len(su_dataloader))

    for sudata, undata in pbar:

        x, y = sudata
        unx, uny = undata

        x, y, unx = x.to(device), y.to(device), unx.to(device)

        optimizer.zero_grad()

        # labeled
        out = net(x)
        loss = loss_func(out.squeeze(), y) / su_batch

        # unlabeled
        out = net(unx)
        with torch.no_grad():
            out_labels = out.argmax(dim=1)

        if epochnum < t2:
            alpha = epochnum / t2 * 1
        else:
            alpha = 1

        loss += loss_func(out.squeeze(), out_labels) * alpha / un_batch

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())

    return np.mean(avg_loss)


def test(net, dataloader, loss_func):
    avg_loss = []
    right = 0
    total = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            out = net(x)
            loss = loss_func(out.squeeze(), y)
            avg_loss.append(loss.item())
            out_labels = out.argmax(dim=1)
            rights = (out_labels==y).float()
            right += rights.sum().item()
            total += out.size(0)

    return np.mean(avg_loss), right/total * 100
