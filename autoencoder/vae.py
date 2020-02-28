import os, tqdm, random, pickle, sys

import torch
import torchvision

from torch import nn
import torch.nn.functional as F
import torch.distributions as ds

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize, Grayscale, Pad
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU
from torch.optim import Adam

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

import util

from torch.utils.tensorboard import SummaryWriter

SEEDFRAC = 2
DV = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):

    def __init__(self, in_size, channels, zs=256, k=3, batch_norm=False):
        super().__init__()

        c, h, w = in_size
        c1, c2, c3, c4, c5 = channels

        # resnet blocks
        self.block1 = util.Block(c,  c1, kernel_size=k, batch_norm=batch_norm)
        self.block2 = util.Block(c1, c2, kernel_size=k, batch_norm=batch_norm)
        self.block3 = util.Block(c2, c3, kernel_size=k, batch_norm=batch_norm)
        self.block4 = util.Block(c3, c4, kernel_size=k, batch_norm=batch_norm)
        self.block5 = util.Block(c4, c5, kernel_size=k, batch_norm=batch_norm)

        self.fs = (h // 2**5) * (w // 2 ** 5) * c5
        self.lin = nn.Linear(self.fs, zs*2)

    def forward(self, x0):

        b = x0.size(0)

        x1 = F.max_pool2d(self.block1(x0), 2)
        x2 = F.max_pool2d(self.block2(x1), 2)
        x3 = F.max_pool2d(self.block3(x2), 2)
        x4 = F.max_pool2d(self.block4(x3), 2)
        x5 = F.max_pool2d(self.block5(x4), 2)

        z = self.lin(x5.view(b, self.fs))

        return z

class Decoder(nn.Module):

    def __init__(self, out_size, channels, outc=1, zs=256, k=3, batch_norm=False):
        super().__init__()

        self.out_size = out_size

        c, h, w = self.out_size
        self.channels = channels
        c1, c2, c3, c4, c5 = self.channels

        # resnet blocks
        self.block5 = util.Block(c5, c4, kernel_size=k, deconv=True, batch_norm=batch_norm)
        self.block4 = util.Block(c4, c3, kernel_size=k, deconv=True, batch_norm=batch_norm)
        self.block3 = util.Block(c3, c2, kernel_size=k, deconv=True, batch_norm=batch_norm)
        self.block2 = util.Block(c2, c1, kernel_size=k, deconv=True, batch_norm=batch_norm)
        self.block1 = util.Block(c1, c,  kernel_size=k, deconv=True, batch_norm=batch_norm)

        self.conv0 = nn.Conv2d(c, c * outc, kernel_size=1)

        self.fs = (h // 2**5) * (w // 2 ** 5) * c5
        self.ss = c5, (h // 2**5), (w // 2 ** 5)
        self.lin = nn.Linear(zs, self.fs)

    def forward(self, z):

        b = z.size(0)
        c, h, w = self.out_size

        x5 = self.lin(z).view(b, self.ss[0], self.ss[1], self.ss[2])

        x4 = F.upsample_bilinear(self.block5(x5), scale_factor=2)
        x3 = F.upsample_bilinear(self.block4(x4), scale_factor=2)
        x2 = F.upsample_bilinear(self.block3(x3), scale_factor=2)
        x1 = F.upsample_bilinear(self.block2(x2), scale_factor=2)
        x0 = F.upsample_bilinear(self.block1(x1), scale_factor=2)

        return self.conv0(x0)

def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    ## Load the data
    if arg.task == 'mnist':
        trainset = torchvision.datasets.MNIST(root=arg.data_dir, train=True,
                                                download=True, transform=ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=arg.data_dir, train=False,
                                               download=True, transform=ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 28, 28

    elif arg.task == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=True,
                                                download=True, transform=ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=False,
                                               download=True, transform=ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 32, 32

    elif arg.task == 'cifar-gs':
        transform = Compose([Grayscale(), ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=arg.data_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 32, 32

    elif arg.task == 'imagenet64':
        transform = Compose([ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 64, 64

    elif arg.task == 'ffhq':
        transform = Compose([ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 128, 128

    else:
        raise Exception('Task {} not recognized.'.format(arg.task))

    zs = arg.latent_size

    encoder = Encoder((C, H, W), arg.channels, zs=zs, k=arg.kernel_size, batch_norm=arg.batch_norm)
    decoder = Decoder((C, H, W), arg.channels, zs=zs, k=arg.kernel_size, batch_norm=arg.batch_norm)

    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=arg.lr)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    instances_seen = 0
    for epoch in range(arg.epochs):

        for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

            # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()

            # -- encoding
            z = encoder(input)

            # -- compute KL losses

            kl_loss = util.kl_loss(z[:, :zs], z[:, zs:])

            # -- take samples
            zsample  = util.sample(z[:, :zs], z[:, zs:])

            # -- decoding
            xout = decoder(zsample)
            rc_loss = F.binary_cross_entropy_with_logits(xout, input, reduction='none').view(b, -1).sum(dim=1)

            loss = rc_loss + arg.beta * kl_loss
            loss = loss.mean()

            instances_seen += input.size(0)

            tbw.add_scalar('style-vae/zkl-loss', float(kl_loss.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/rec-loss', float(rc_loss.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/total-loss', float(loss.data.item()), instances_seen)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(encoder.state_dict(), './encoder.model')
        torch.save(decoder.state_dict(), './decoder.model')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task: [mnist, cifar10].",
                        default='mnist', type=str)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-c", "--channels",
                        dest="channels",
                        help="Number of channels per block (list of 5 integers).",
                        nargs=5,
                        default=[32, 64, 128, 256, 512],
                        type=int)

    parser.add_argument("--batch-norm",
                        dest="batch_norm",
                        help="Adds batch normalization after each block.",
                        action='store_true')

    parser.add_argument("--evaluate-every",
                        dest="eval_every",
                        help="Run an exaluation/sample every n epochs.",
                        default=1, type=int)

    parser.add_argument("-k", "--kernel_size",
                        dest="kernel_size",
                        help="Size of convolution kernel",
                        default=3, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-z", "--latent-size",
                        dest="latent_size",
                        help="Size of latent space.",
                        default=256, type=int)

    parser.add_argument('--beta',
                        dest='beta',
                        help="Beta param.",
                        type=float,
                        default=1.0)

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit on the number of instances seen per epoch (for debugging).",
                        default=None, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate.",
                        default=0.0005, type=float)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/style', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)