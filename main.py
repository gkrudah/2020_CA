import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse
import os

from config import device
from model import ClassficationModel
from train import train, mix_train, test
from evaluate import eval


def split_data(classes, dataset, end, start=0):
    datalist = [[] for _ in range(len(classes))]
    newdatalist = [[] for _ in range(len(classes))]

    for i in range(len(dataset)):
        datalist[dataset[i][1]].append(i)

    for i in range(len(datalist)):
        newdatalist[i] = datalist[i][int(len(datalist[i]) * start):int(len(datalist[i]) * end)]

    return newdatalist


def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--saved_model', type=str, default='71.84_type1.pth')
    # parser.add_argument()

    # parameter
    parser.add_argument('--type', type=int, default=3, help='type must be 1, 2, 3')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=150)

    args = parser.parse_args()

    if not os.path.exists(args.data):
        os.makedirs(args.data)

    # supervised dataset will be 1/2
    supervised_ratio = 0.5
    ratio = 0.25 * (1 + args.type)

    pretrain = False
    lr = args.lr
    if args.type == 1:
        pretrain = True
        # lr *= 10
    elif args.type == 2:
        unbatch = int(args.batch / 2)
    elif args.type == 3:
        unbatch = args.batch

    # preparing data
    transform = transforms.Compose([transforms.Resize(32, 32), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.mode == 'train':
        trainset = torchvision.datasets.CIFAR10(root=args.data + '/train', train=True, download=True,
                                                transform=transform)
        classes = trainset.classes

        # for splitting dataset, make dataset labeled and unlabeled
        supervised_traindata = split_data(classes, trainset, supervised_ratio)

        supervised_idxlist = []
        for data in supervised_traindata:
            supervised_idxlist += data

        splittrainset = torch.utils.data.Subset(trainset, supervised_idxlist)
        su_trainloader = torch.utils.data.DataLoader(dataset=splittrainset, batch_size=args.batch, shuffle=True,
                                                     num_workers=2, drop_last=True)

        if not pretrain:
            un_traindata = split_data(classes, trainset, end=ratio, start=supervised_ratio)

            idxlist = []
            for data in un_traindata:
                idxlist += data

            splittrainset = torch.utils.data.Subset(trainset, idxlist)
            un_trainloader = torch.utils.data.DataLoader(dataset=splittrainset, batch_size=unbatch, shuffle=True,
                                                         num_workers=2, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root=args.data + '/test', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch, shuffle=True, num_workers=2,
                                             drop_last=True)

    # prepare net
    classes = 10
    net = ClassficationModel(classes)
    net = net.to(device)
    net = nn.DataParallel(net)

    if not pretrain:
        net.load_state_dict(torch.load(os.path.join(args.checkpoint, args.saved_model)))

    if args.mode == 'train':
        bestnet = ClassficationModel(classes)
        bestnet = bestnet.to(device)
        bestnet = nn.DataParallel(bestnet)

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.8)

    # main activity
    bestacc = 0
    patience = 0

    if args.mode == 'train':
        for epoch in range(1, args.epoch + 1):

            net.train()
            if pretrain:
                train_loss = train(net, su_trainloader, optimizer, loss_func)
            else:
                train_loss = mix_train(net, su_trainloader, un_trainloader, optimizer, loss_func, epoch, args.batch, unbatch)

            print('Train:', epoch, train_loss)

            net.eval()
            test_loss, test_acc = test(net, testloader, loss_func)
            print('Test:', epoch, test_loss, str(test_acc) + '%')

            if bestacc < test_acc:
                print('NEW BEST MODEL')
                bestacc = test_acc
                bestnet.load_state_dict(net.state_dict())
                patience = 0

            if patience >= args.patience:
                print('EARLY STOP')
                break
            else:
                patience += 1

        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

        torch.save(bestnet.state_dict(), os.path.join(args.checkpoint, str(bestacc)[:5] + '_type' + str(args.type) + '.pth'))

    elif args.mode == 'eval':
        net.eval()
        testacc = eval(net, testloader)
        print('Test acc:', str(testacc) + '%')


if __name__ == "__main__":
    main()