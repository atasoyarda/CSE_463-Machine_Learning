"""
total parameters for F1
784x64+64x10+64+10=50890

total parameters for F2
784x64+64x64+64x32+64+32+10=52650
"""
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
import torch.nn as Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

def load_dataset(dataset: str, small: bool = False):

        with np.load( "mnist.npz", allow_pickle=True) as f:
            X_train, labels_train = f['x_train'], f['y_train']
            X_test, labels_test = f['x_test'], f['y_test']

        # Reshape each image of size 28 x 28 to a vector of size 784
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

        # Pixel values are integers from 0 to 255.
        # Dividing the pixel values by 255 is a technique called normalization,
        # which is a standard practice in Machine Learning to prevent large numeric values
        X_train = X_train / 255
        X_test = X_test / 255

        return ((torch.from_numpy(X_train).float(), torch.from_numpy(labels_train)),
                  (torch.from_numpy(X_test).float(), torch.from_numpy(labels_test)))

def preapare_dataset():
    (x, y), (x_test, y_test) = load_dataset("mnist")
    train_loader = DataLoader([[x[i], y[i]] for i in range(len(y))], shuffle=True, batch_size=100)
    test_loader = DataLoader([ [x_test[i], y_test[i]] for i in range(len(y_test))], shuffle=True, batch_size=100)
    return train_loader, test_loader

class F1(Module.Module):
    def __init__(self, h0: int, d: int, k: int):
        super().__init__()
        self.model = Module.Sequential(
            Module.Linear(d,h0),
            Module.ReLU(),
            Module.Linear(h0,k),
        )

    def forward(self, x):
        x = self.model(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class F2(Module.Module):

    def __init__(self, h0: int, h1: int, d: int, k: int):
        super().__init__()

        self.model = Module.Sequential(
            Module.Linear(d,h0),
            Module.ReLU(),
            Module.Linear(h0,h1),
            Module.ReLU(),
            Module.Linear(h1,k),
        )

    def forward(self, x):
        x = self.model(x)
        return torch.nn.functional.log_softmax(x, dim=1)

train_loader, test_loader = preapare_dataset()

def train(model: Module, optimizer: Adam, train_loader: DataLoader):
    model.train()
    loss_history = []
    last_accuracy = 0
    epochs = 0
    while last_accuracy/ len(train_loader.dataset) <= 0.99:
        loss_epoch = 0
        last_accuracy = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            forward = model(x)
            preds = torch.argmax(forward, 1)
            last_accuracy += torch.sum(preds == y).item()
            loss = cross_entropy(forward, y)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
        epochs+=1
        print("Epoch-", epochs, "Loss: ", loss_epoch / len(train_loader.dataset),
              "Accuracy: ", last_accuracy/ len(train_loader.dataset))
        loss_history.append(loss_epoch / len(train_loader.dataset))
    return loss_history

if __name__ == '__main__':

    model1 = F1(64,784,10)
    optimizer1 = Adam(model1.parameters(), lr=1e-3)
    losses=train(model1,optimizer1,train_loader)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses, '--x', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('F1.png')
    loss_epoch = 0
    last_accuracy = 0
    for x, y in test_loader:
        forward = model1(x)
        preds = torch.argmax(forward, 1)
        last_accuracy += torch.sum(preds == y).item()
        loss = cross_entropy(forward, y)
        loss_epoch += loss.item()
    print('Testing dataset')
    print("Loss:", loss_epoch / len(test_loader))
    print("Accuracy:", last_accuracy / len(test_loader.dataset))

    model2 = F2(64,32,784,10)
    optimizer2 = Adam(model2.parameters(), lr=1e-3)
    losses=train(model2,optimizer2,train_loader)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses, '--x', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('F2.png')
    loss_epoch = 0
    last_accuracy = 0
    for x, y in test_loader:
        forward = model2(x)
        preds = torch.argmax(forward, 1)
        last_accuracy += torch.sum(preds == y).item()
        loss = cross_entropy(forward, y)
        loss_epoch += loss.item()
    print('Testing dataset')
    print("Loss:", loss_epoch / len(test_loader))
    print("last_accuracy:", last_accuracy / len(test_loader.dataset))