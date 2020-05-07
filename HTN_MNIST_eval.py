# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
# from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import matplotlib.pyplot as plt
#import tncontract as tn
import numpy as np
from itertools import product
from itertools import combinations
import scipy.io as sio
import time
# torch.manual_seed(1)  # reproducible

from TN_FC_MNIST_pytorch_GPU import TensorLayer
from TN_FC_MNIST_pytorch_GPU import TTN


#data_folder = "/export/tree_tn/data/mnist/"
GPU_flag=1
if GPU_flag ==1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
ttn = TTN().to(device)

model_file = 'TNCF_checkpoint.pth'
if os.path.isfile(model_file):
    checkpoint = torch.load(model_file)
    accuracy = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']
    ttn.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # losslist = checkpoint['loss_list']
    bond_data = checkpoint['bond_data']
    bond_inner = checkpoint['bond_inner']
    print('Load checkpoint at epoch %d | accuracy %.2f | bond_data %d | bond_inner %d', (start_epoch, accuracy, bond_data, bond_inner))


def feature_map(data28, bond_data):
    N = data28.size()[0]
    data32 = torch.zeros(N, 32,32)
    data32[:, 2:30, 2:30]=data28

    data_group = torch.zeros(N, 32, 32, bond_data)
    # pi = torch.Tensor([np.pi])
    for i in range(bond_data):
        data_group[:, :, :, i] = (len([c for c in combinations(range(bond_data - 1), i)]) ** 0.5) * \
                                    torch.cos((data32/255) * (3.1416 / 4)) ** (bond_data - (i + 1)) * torch.sin(
            (data32/255) * (3.1416 / 4)) ** i
    return data_group


# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_data = torchvision.datasets.FashionMNIST(root='./fmnist/', train=False)
testx = feature_map(test_data.data, bond_data)
test_x = Variable(testx, requires_grad=True)
test_l = test_data.targets


def main():
        N=2
        M=int(10000/N)
        sum=0
        for i in range(N):
            text_single=test_x[M * i:M * i + M, :, :, :].to(device)
            lable_single=test_l[M * i:M * i + M].to(device)
            test_output = ttn(text_single)[0]
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            sum = int(torch.sum(pred_y == lable_single))+sum
        accuracy = sum/10000
        checkpoint['accuracy'] = accuracy
        torch.save(checkpoint, model_file)
        print('test accuracy: %.2f' % accuracy)


if __name__ == '__main__':
    main()