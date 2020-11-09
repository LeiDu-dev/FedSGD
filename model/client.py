import torch

from torch import nn
from torch.autograd import Variable

from model.lenet import lenet5


class client(object):
    def __init__(self, rank, data_loader):
        # seed
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # rank
        self.rank = rank

        # data loader
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

    @staticmethod
    def __load_global_model():
        global_model_state = torch.load('./cache/global_model_state.pkl')
        model = lenet5().cuda()
        model.load_state_dict(global_model_state)
        return model

    def __train(self, model):
        train_loss = 0
        train_correct = 0
        model.train()
        for data, target in self.train_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            train_loss += loss
            loss.backward()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        grads = {'n_samples': len(self.train_loader.dataset), 'named_grads': {}}
        for name, param in model.named_parameters():
            grads['named_grads'][name] = param.grad

        print('[Rank {:>2}]  Loss: {:>4.6f},  Accuracy: {:>.4f}'.format(
            self.rank,
            train_loss,
            train_correct / len(self.train_loader.dataset)
        ))
        return grads

    def run(self):
        model = self.__load_global_model()
        grads = self.__train(model=model)
        torch.save(grads, './cache/grads_{}.pkl'.format(self.rank))
