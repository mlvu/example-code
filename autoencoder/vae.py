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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter

SEEDFRAC = 2
DV = 'cuda' if torch.cuda.is_available() else 'cpu'

RED  = '#C82506'
BLUE = '#0365C0'
GREEN = '#00882B'
ORANGE = '#DE6A10'
PURPLE = '#773F9B'
YELLOW = '#DCBD23'

# workaround for weird bug (macos only)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

OUT_EPOCHS = [0, 1, 2, 3, 4, 5, 10, 25, 50, 100, 150, 200, 250, 300]

def gather(generator, batches, numpy=True):
    res = []
    for i, batch in enumerate(generator):
        res.append(batch[0])
        if i > batches:
            break

    res = torch.cat(res, axis=0)
    
    if numpy:
        return res.permute(0, 2, 3, 1).cpu().data.numpy()
    return res

class Encoder(nn.Module):

    def __init__(self, in_size, channels, zs=256, k=3, batch_norm=False, vae=False):
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
        self.lin = nn.Linear(self.fs, zs*(2 if vae else 1))

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
    grayscale = False

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
                                                  shuffle=False, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 64, 64

    elif arg.task == 'ffhq':

        # These are people in the data that smile
        SMILING = [1, 6, 7, 9, 14,   16, 17, 20, 21, 29,    30, 31, 33, 40, 45,    51, 55, 57, 58, 61]
        NONSMILING = [0, 2, 3, 4, 5,    8, 10, 11, 18, 19,     23, 25, 27, 28, 32,    34, 35, 36, 37, 44]

        transform = Compose([ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=False, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data_dir+os.sep+'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 128, 128
        
        grayscale = False

    else:
        raise Exception('Task {} not recognized.'.format(arg.task))


    # extract the first 500 faces into a tensor
    faces = gather(trainloader, 500 // arg.batch_size)
    facespt = gather(trainloader, 500 // arg.batch_size, numpy=False)

    if torch.cuda.is_available():
        facespt = facespt.cuda()

    # plot data
    fig = plt.figure(figsize=(5, 20))
    for i in range(5 * 20):
        ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces[i] * (np.ones(3) if grayscale else 1))
        ax.set_title(i)

    plt.tight_layout()
    plt.savefig('faces.pdf')

    # plot data
    fig = plt.figure(figsize=(5, 20))
    for i in range(5 * 20):
        ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces[i] * (np.ones(3) if grayscale else 1))

    plt.tight_layout()
    plt.savefig('faces-notitle.pdf')

    # smiling/nonsmiling
    fig = plt.figure(figsize=(5, 4))
    for i in range(len(SMILING)):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces[SMILING[i]] * (np.ones(3) if grayscale else 1))

    plt.savefig('smiling-faces.pdf')

    fig = plt.figure(figsize=(5, 4))
    for i in range(len(NONSMILING)):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces[NONSMILING[i]] * (np.ones(3) if grayscale else 1))

    plt.savefig('nonsmiling-faces.pdf')

    zs = arg.latent_size

    encoder = Encoder((C, H, W), arg.channels, zs=zs, k=arg.kernel_size, batch_norm=not arg.no_batch_norm, vae=arg.variational)
    decoder = Decoder((C, H, W), arg.channels, zs=zs, k=arg.kernel_size, batch_norm=not arg.no_batch_norm)

    if arg.cp is not None:
        encoder.load_state_dict(torch.load(arg.cp + os.sep + 'encoder.model', map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(arg.cp + os.sep + 'decoder.model', map_location=torch.device('cpu')))

    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=arg.lr)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    instances_seen = 0
    for epoch in range(arg.epochs):

        if epoch in OUT_EPOCHS:
            with torch.no_grad():

                v = 'vae' if arg.variational else 'ae'
                torch.save(encoder.state_dict(), f'./encoder.{v}.{epoch:04}.model')
                torch.save(decoder.state_dict(), f'./decoder.{v}.{epoch:04}.model')

                # reconstructions

                # plot reconstructions
                lts = encoder(facespt[:5 * 20])
                rec = torch.sigmoid(decoder(lts[:, :arg.latent_size]))
                rec = rec.permute(0, 2, 3, 1).cpu().data.numpy()

                fig = plt.figure(figsize=(5, 20))
                for i in range(5 * 20):
                    ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
                    ax.imshow(rec[i] * (np.ones(3) if grayscale else 1))

                plt.tight_layout()
                plt.savefig(f'reconstructions.{epoch:04}.pdf')

                # random samples
                lts = torch.randn(100, arg.latent_size, device=DV)
                outs = torch.sigmoid(decoder(lts))
                outs = outs.permute(0, 2, 3, 1).cpu().data.numpy()

                fig = plt.figure(figsize=(5, 20))
                for i in range(5 * 20):
                    ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
                    ax.imshow(outs[i] * (np.ones(3) if grayscale else 1))

                plt.tight_layout()
                plt.savefig(f'samples.{epoch:04}.pdf')


        for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

            # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()

            # -- encoding
            z = encoder(input)

            if arg.variational:
                kl_loss = util.kl_loss(z[:, :zs], z[:, zs:])
                zsample  = util.sample(z[:, :zs], z[:, zs:])
            else:
                kl_loss = 0
                zsample = z

            # -- decoding
            xout = decoder(zsample)
            rc_loss = F.binary_cross_entropy_with_logits(xout, input, reduction='none').view(b, -1).sum(dim=1)

            loss = rc_loss + arg.beta * kl_loss
            loss = loss.mean()

            instances_seen += input.size(0)

            if arg.variational:
                tbw.add_scalar('style-vae/zkl-loss', float(kl_loss.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/rec-loss', float(rc_loss.data.mean(dim=0).item()), instances_seen)
            tbw.add_scalar('style-vae/total-loss', float(loss.data.item()), instances_seen)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task: [mnist, cifar10, ffhq].",
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

    parser.add_argument("--no-batch-norm",
                        dest="no_batch_norm",
                        help="No batch normalization after each block.",
                        action='store_true')

    parser.add_argument("--variational",
                        dest="variational",
                        help="Uses a VAE instead of a regular AE.",
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

    parser.add_argument("--checkpoint",
                        dest="cp",
                        help="Checkpoint directory",
                        default=None, type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/style', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)