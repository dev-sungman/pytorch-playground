### reference : https://discuss.pytorch.org/t/visualize-feature-map/29597

""" import libraries """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torhcvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt
import argparse

import sys
import os

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=1)

    return parser.parse_args()

def main(args):
    """ typical pytorch flow: dataset, loss, model setting """
    train_transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    dataset = datasets.MNIST(root='data', transform=train_transform)

    loader = DataLoader(dataset, num_workers=2, batch_size=8, shuffle=True, pin_memory=True)
    
    if args.models == 'vgg16':
        model = models.vgg16(num_classes=10)
    else:
        model = models.resnet50(num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    """ function for get activation """
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for epoch in range(args.epochs):
        for data, target in loader:
            """ find module in your model """ 
            print(model)

            """ before input your data into model, you have to register forward hook """
            model.layer1.register_forward_hook(get_activation('layer1'))
            # model.features.register_forward_hook(get_activation('features')) # if vgg
            
            output = model(data)

            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            act = activation['layer1'].squeeze()
            # act = activation['features'].squeeze() # if vgg
            
            """ for visualization using matplotlib """
            fig, axarr = plt.subplots(act.size(0))
            for idx in range(act.size(0)):
                axarr[idx].imshow(act[idx])


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
