import torch

from matplotlib import pyplot as plt


def plot():
    accuracy = torch.load('../cache/accuracy.pkl')
    plt.plot([e for e in range(1, len(accuracy) + 1)], accuracy, label='FedSGD')

    plt.title("Test Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.ylim(0, 1)
    plt.xlim(1, len(accuracy))
    plt.legend(loc=4)

    plt.show()
