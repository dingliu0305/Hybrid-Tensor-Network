import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as Data
import os
from itertools import product
from itertools import combinations

if not os.path.exists('./TN_ImageCopresssimg'):
    os.mkdir('./TN_ImageCopresssimg')
#
#
def to_img(x):
    # out = 0.5 * (x + 1)
    # out = out.clamp(0, 1)
    # out = out.view(-1, 1, 28, 28)

    out = x.view(-1, 1, 28, 28)
    return out


batch_size = 24
num_epoch = 20000
bond_data = 2
bond_inner = 2
DIM = 32



GPU_flag = 1
loadcheckpoint = 0
train_flag = 1

if GPU_flag ==1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


train_data = torchvision.datasets.MNIST(
# train_data = torchvision.datasets.FashionMNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
)
train_data.data = train_data.data[0:batch_size, :, :]
dataloader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()





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



# Compressor
class Compressor(nn.Module):
    def __init__(self):
        super(Compressor, self).__init__()
        self.tensorL1 = nn.Sequential(TensorLayer(16, bond_inner, bond_data, batch_size, 0))
                                      # nn.Tanh())
        self.tensorL2 = nn.Sequential(TensorLayer(8,  1, bond_inner, batch_size, 1))
        self.tensorL3 = nn.Sequential(TensorLayer(4,  bond_inner, bond_inner, batch_size, 0))
        self.tensorL4 = nn.Sequential(TensorLayer(2,  1, bond_inner, batch_size, 1))
        self.tensorL5 = nn.Sequential(TensorLayer(1, 784, bond_inner, batch_size, 1))

        self.fc1 = nn.Linear(4*4, 1000)
        self.fc2 = nn.Linear(1000,784)


        self.decoder88 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 4, stride=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 8, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=1, padding=1),  # b, 1, 28, 28
            nn.Tanh())

        self.decoder44 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 5, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 8, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=1, padding=1),  # b, 1, 28, 28
            nn.Tanh())

        self.decoder22 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 5, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 8, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=1),  # b, 1, 28, 28
            nn.Tanh())


    def feature_map(self, data28):
        N = data28.size()[0]
        data32 = torch.zeros(N, 32, 32)
        data28 =data28.squeeze(1)
        data32[:, 2:30, 2:30] = data28
        data_group = torch.zeros(N, 32, 32, bond_data)
        for i in range(bond_data):
            data_group[:, :, :, i] = (len([c for c in combinations(range(bond_data - 1), i)]) ** 0.5) * \
                                     torch.cos((data32) * (3.1416 / 4)) ** (bond_data - (i + 1)) * torch.sin(
                (data32) * (3.1416 / 4)) ** i
        return data_group

    def forward(self, x):
        x = self.feature_map(x)
        x = self.tensorL1(x)
        x = self.tensorL2(x)
        # x = self.tensorL3(x)
        # x = self.tensorL4(x)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = self.tensorL5(x)

        compressed = x
        # x = self.measureL(x)

        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        x = x.permute(0, 3, 1, 2)
        x = self.decoder88(x)
        x = x.view(x.size(0), -1)

        return x, compressed  # return x for visualization


Comp = Compressor()
if torch.cuda.is_available():
    Comp = Comp.cuda()

lossfunc = nn.MSELoss()
optimizer = torch.optim.Adam(Comp.parameters(), lr=0.0005)
# optimizer = torch.optim.RMSprop(Comp.parameters(), lr=0.0005)



# Start training

if loadcheckpoint == True:
    model_file = 'TN_ImageCompress_checkpoint.pth'
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        start_epoch = checkpoint['epoch']
        Comp.load_state_dict(checkpoint['model_Comp'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch %d :' % (start_epoch))
else:
    start_epoch = 0


# z = Variable(torch.rand(batch_size, 16, 16,2)).cuda()

if train_flag == 1:
    for epoch in range(start_epoch, num_epoch):
        for i, (img,_) in enumerate(dataloader):
            print(i)
            num_img = batch_size
            output, compressed1 = Comp(img)  # cnn output
            img_t = img.view(img.size(0), -1).cuda()
            loss = lossfunc(output, img_t)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            if (i) % 10 == 0:
                print('Epoch [{}/{}], loss: {:.6f}, '.format(
                    epoch, num_epoch, loss.item()))
        checkpoint = {
            'epoch': epoch,
            'model_Comp': Comp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'bond_data': bond_data,
            'bond_inner': bond_inner
        }

        model_file = 'TN_ImageCompress_checkpoint.pth'
        torch.save(checkpoint, model_file)


else:
    for i, (img, _) in enumerate(dataloader):

        input_images = to_img(img.cpu().data)
        save_image(input_images, './TN_ImageCopresssimg/input_images.JPG')

        output, compressed= Comp(img)  # cnn output
        img = img.view(img.size(0), -1).cuda()
        psnr = torch.zeros(batch_size)
        for k in range(batch_size):
            t = output-img
            mse = torch.sum((output[k, :]-img[k, :])**2/(28*28))
            psnr[k] = 10 * torch.log10(1/mse)
        psnr_ave = torch.mean(psnr)

    print("psnr_ave:", psnr_ave)
    print(psnr)
    images_compressed = compressed.permute(0,3,1,2).cpu().data
    save_image(images_compressed, './TN_ImageCopresssimg/compressed.JPG')

    output_images = to_img(output.cpu().data)
    save_image(output_images, './TN_ImageCopresssimg/output_images.png')