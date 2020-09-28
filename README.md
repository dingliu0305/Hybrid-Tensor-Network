Welcome to star this projects, it will be helpful for you to learn tensor network caculating in Pytorch.

This is the computational appendix for the following paper:

Ding Liu, Zekun Yao, Quan Zhang. Quantum-Classical Machine learning by Hybrid Tensor Networks. arXiv:2005.09428v1, 2020.

# Files

* HTN_MNIST：The baseline for fashion-MNIST dataset.If you want to train the MNIST dataset, you can just modify the `torchvision.datasets.FashionMNIST` to `torchvision.datasets.MNIST`。The  best HTN result 90% on fashion-MNIST will come true.And you can also just load the model from `great_checkpoint/fashionmnist-90%ACC_checkpoint.pth` to test.
* HTN_MNIST_V2：Generating loss curve figure and Adding tensorboard to monitor the loss and acc.
* HTN_MNIST_CNN_V3：Modifing the Neural network structure with CNN.
* HTN_MNIST_eval：Loading the `.pth` type file to evaluate the fashion-MNIST(or MNIST)，and you will get an accuracy about testing.
* HTN_AutoEncoder.py：Use the training model from`.pth` file to encode the images.