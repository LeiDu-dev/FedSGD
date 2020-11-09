import torch

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


class loader(object):
    def __init__(self, cmd='cifar10', batch_size=64):
        self.cmd = cmd
        self.batch_size = batch_size
        self.__load_dataset()
        self.__get_index()

    def __load_dataset(self):
        # mnist
        self.train_mnist = datasets.MNIST('./dataset/',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

        self.test_mnist = datasets.MNIST('./dataset/',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))

        # cifar10
        self.train_cifar10 = datasets.CIFAR10('./dataset/',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                              ]))
        self.test_cifar10 = datasets.CIFAR10('./dataset/',
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                             ]))

    def __get_index(self):
        if self.cmd == 'cifar10':
            self.train_dataset = self.train_cifar10
            self.test_dataset = self.test_cifar10
        else:
            self.train_dataset = self.train_mnist
            self.test_dataset = self.test_mnist

        self.indices = [[], [], [], [], [], [], [], [], [], []]
        for index, data in enumerate(self.train_dataset):
            self.indices[data[1]].append(index)

    def get_loader(self, rank):
        dataset_indices = []
        difference = list(set(range(10)).difference(set(rank)))
        for i in difference:
            dataset_indices.extend(self.indices[i])

        dataset = torch.utils.data.Subset(self.train_cifar10, dataset_indices)
        if self.cmd != 'cifar10':
            dataset = torch.utils.data.Subset(self.train_mnist, dataset_indices)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader
