import torch

from torch import optim
from torch.autograd import Variable

from model.lenet import lenet5


class server(object):
    def __init__(self, size, data_loader):
        self.size = size
        self.test_loader = data_loader[1]
        self.path = './cache/global_model_state.pkl'
        self.model = self.__init_server()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.accuracy = []

    def __init_server(self):
        model = lenet5().cuda()
        torch.save(model.state_dict(), self.path)
        return model

    def __load_grads(self):
        grads_info = []
        for s in range(self.size):
            grads_info.append(torch.load('./cache/grads_{}.pkl'.format(s)))
        return grads_info

    @staticmethod
    def __average_grads(grads_info):
        total_grads = {}
        n_total_samples = 0
        for info in grads_info:
            n_samples = info['n_samples']
            for k, v in info['named_grads'].items():
                if k not in total_grads:
                    total_grads[k] = v
                total_grads[k] += v * n_samples
            n_total_samples += n_samples
        gradients = {}
        for k, v in total_grads.items():
            gradients[k] = torch.div(v, n_total_samples)
        return gradients

    def __step(self, gradients):
        self.model.train()
        self.optimizer.zero_grad()
        for k, v in self.model.named_parameters():
            v.grad = gradients[k]
        self.optimizer.step()

    def __test(self):
        test_correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data).cuda(), Variable(target).cuda()
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        return test_correct / len(self.test_loader.dataset)

    def aggregate(self):
        grads_info = self.__load_grads()
        gradients = self.__average_grads(grads_info)

        self.__step(gradients)
        torch.save(self.model.state_dict(), './cache/global_model_state.pkl')

        test_accuracy = self.__test()
        self.accuracy.append(test_accuracy)
        torch.save(self.accuracy, './cache/accuracy.pkl')
        print('\n[Global Model]  Test Accuracy: {:.2f}%\n'.format(test_accuracy * 100.))
