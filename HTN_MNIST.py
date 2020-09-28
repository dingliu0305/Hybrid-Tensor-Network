# library
# standard library
import os
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


# Hyper Parameters
EPOCH = 170  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1024
LR = 0.001  # learning rate

#data_folder = "/export/tree_tn/data/mnist/"

# data_dimention=32
bond_data = 3
bond_inner = 3
# bond_top =20
# n_train_each=5000
# n_test_each = 800
n_train = 60000


loadcheckpoint = 0
GPU_flag = 0

if GPU_flag ==1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def feature_map(data28, bond_data):
    N = data28.size()[0]
    # data32 = data.float()
    data32 = torch.zeros(N, 32,32)
    data32[:, 2:30, 2:30]=data28

    data_group = torch.zeros(N, 32, 32, bond_data)
    # pi = torch.Tensor([np.pi])
    for i in range(bond_data):
        data_group[:, :, :, i] = (len([c for c in combinations(range(bond_data - 1), i)]) ** 0.5) * \
                                    torch.cos((data32/255) * (3.1416 / 4)) ** (bond_data - (i + 1)) * torch.sin(
            (data32/255) * (3.1416 / 4)) ** i

    return data_group


# train_data = torchvision.datasets.MNIST(
train_data = torchvision.datasets.FashionMNIST(
    './fmnist/', train=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None, download=True)
trainx = feature_map(train_data.data[0:n_train,:,:], bond_data)
trainl = torch.LongTensor(train_data.targets[0:n_train].numpy())
datatensor = Data.TensorDataset(trainx, trainl) # wrap trainx and trainl by Data.TensorDataset
train_loader = Data.DataLoader(dataset=datatensor, batch_size=BATCH_SIZE, shuffle=True) # wrap datatensor by Data.DataLoader

# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# testx = feature_map(test_data.data, bond_data)
# test_x = Variable(testx, requires_grad=True)
# test_l = test_data.targets


class TensorLayer(nn.Module):

    def __init__(self, tn_size, bond_up, bond_down, batchsize, top_flag):
        super(TensorLayer, self).__init__()
        self.bond_up = bond_up
        self.bond_down = bond_down
        self. tn_size = tn_size
        self. bacthsize = batchsize
        self. top_flag = top_flag
        self.weight = nn.Parameter(
            data=Variable(torch.rand(self. tn_size, self. tn_size, self.bond_down, self.bond_down, self.bond_down, self.bond_down, self.bond_up),
                          requires_grad=True), requires_grad=True)



    def forward(self,data):
        Num = data.size(0)
        output = Variable(torch.zeros(Num,  self. tn_size, self. tn_size, self.bond_up)).to(device)
        for m, n in product(range(self.tn_size), range(self.tn_size)):
            bond = data[:, 2*m,2*n,:].shape[1]
            local_result = torch.zeros(Num, bond, bond, bond, bond).to(device)
            for i, j, k, l in product(range(bond), range(bond), range(bond), range(bond)):
                local_result[:, i, j, k, l] = data[:,  2*m,2*n,i] * data[:,  2*m,2*n+1,j] * data[:,  2*m+1,2*n,k] * data[:, 2*m+1,2*n+1,l]

            temp = torch.einsum('mijkl, ijkln -> mn', local_result, self.weight[m,n,:,:,:,:,:])
            if self.top_flag == 0:
                output[:,  m, n, :] = torch.nn.functional.normalize(temp, dim=1, p=2)
            else:
                output[:,  m, n, :]=temp
        return output


class TTN(nn.Module):
    def __init__(self):
        super(TTN, self).__init__()
        self.tensorL1 = nn.Sequential(TensorLayer(16, bond_inner, bond_data, BATCH_SIZE, 0))
        self.tensorL2 = nn.Sequential(TensorLayer(8,  bond_inner, bond_inner, BATCH_SIZE, 0))
        self.tensorL3 = nn.Sequential(TensorLayer(4,  bond_inner, bond_inner, BATCH_SIZE, 0))
        self.tensorL4 = nn.Sequential(TensorLayer(2,  6, bond_inner, BATCH_SIZE, 1))
        self.tensorL5 = nn.Sequential(TensorLayer(1, 1, bond_inner, BATCH_SIZE, 1))

        self.fc1 = nn.Linear(2*2*6, 1000)
        self.fc2 = nn.Linear(1000, 10)



    def forward(self, x):
        x = self.tensorL1(x)
        x = self.tensorL2(x)
        x = self.tensorL3(x)
        x = self.tensorL4(x)
        quantum_layer4 = x
        # x = self.tensorL5(x)
        # output = x[:,0,0,0,:]
        x = x.view(x.size(0), -1)
        # x = F.relu(self.bn(x))
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        # output = x.squeeze()
        # return F.log_softmax(output, dim=1), quantum_layer4  # return x for visualization
        return output, quantum_layer4

def main():
    ttn = TTN().to(device)
    optimizer = torch.optim.Adam(ttn.parameters(), lr=LR)  # optimize all cnn parameters, if you just want to optimize tensorL1, please use ttn.tensorL1.parameters()
    loss_func = nn.CrossEntropyLoss()

    plt.ion()
    # training and testing

    if loadcheckpoint == True:
        model_file = 'TNCF_checkpoint.pth'
        if os.path.isfile(model_file):
            checkpoint = torch.load(model_file)
            accuracy = checkpoint['accuracy']
            start_epoch = checkpoint['epoch']
            ttn.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            losslist = checkpoint['loss_list']
            print('Load checkpoint at epoch %d | accuracy %.2f' % (start_epoch, accuracy))
    else:
        start_epoch = 0
        accuracy = 0

    for epoch in range(start_epoch, EPOCH):
        for step, (x, y) in enumerate(train_loader): # gives batch data, normalize x when iterate train_loader
            starttime = time.time()

            b_x = Variable(x).to(device)  # batch x
            b_y = Variable(y) .to(device) # batch y

            output = ttn(b_x)[0] # cnn output
            pred_y = torch.max(output,1)[1]
            accuracy = int(torch.sum(b_y == pred_y)) / BATCH_SIZE
            # quantum_layer = ttn(b_x)[1].detach().cpu()
            # plt.imshow(quantum_layer[0, :, :,0], cmap='gray')
            # plt.show()
            # plt.pause(0.1)

            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
            checkpoint = {
                'accuracy': accuracy,
                'epoch': epoch,
                'model': ttn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_list': loss,
                'bond_data': bond_data,
                'bond_inner': bond_inner
            }
            model_file = 'TNCF_checkpoint.pth'
            torch.save(checkpoint, model_file)

    plt.ioff()

if __name__ == '__main__':
    main()