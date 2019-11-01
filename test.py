from functional_warm_up import FuncLRScheduler
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


def linear(last_epoch, lr_goal, warm_epochs):
    if last_epoch > warm_epochs:
        return -1
    return last_epoch * lr_goal / warm_epochs


def quad_square(last_epoch, lr_goal, warm_epochs):
    if last_epoch > warm_epochs:
        return -1
    return math.sqrt(last_epoch) * lr_goal / math.sqrt(warm_epochs)


def quadratic(last_epoch, lr_goal, warm_epochs):
    if last_epoch > warm_epochs:
        return -1
    return (last_epoch ** 2) * lr_goal / (warm_epochs ** 2)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.conv3 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


if __name__ == '__main__':
    lr = []
    epochs = []

    net = Model()

    params = [{'params': net.conv1.parameters(), 'weight_decay': 0.1},
              {'params': net.conv2.parameters(), 'weight_decay': 0.2},
              {'params': net.conv3.parameters(), 'weight_decay': 0.3}]
    opt = optim.Adam(params=params)
    n_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    lr_goal = [0.1, 0.3, 0.5]
    fns = [quadratic, linear, quad_square]
    wu_scheduler = FuncLRScheduler(optimizer=opt, lr_goal=lr_goal, warm_epochs=[30, 10, 70], fns=fns,
                                   scheduler_after=n_scheduler)

    epochs_before_save = 10
    for epoch in range(0, epochs_before_save):
        for param_group in opt.param_groups:
            epochs.append(epoch)
            lr.append(param_group['lr'])
            print("epoch:", epoch, "lr:", round(param_group['lr'], 5), "weight_decay:", param_group['weight_decay'])
        wu_scheduler.step()

    torch.save(opt.state_dict(), "opt_state.pt")
    opt.load_state_dict(torch.load("opt_state.pt"))

    for epoch in range(0, 100):
        for param_group in opt.param_groups:
            epochs.append(epoch + epochs_before_save)
            lr.append(param_group['lr'])
            print("epoch:", epoch + epochs_before_save, "lr:", round(param_group['lr'], 5),
                  "weight_decay:", param_group['weight_decay'])
        wu_scheduler.step()

    for i in range(len(lr_goal)):
        plt.plot(epochs[i::len(lr_goal)], lr[i::len(lr_goal)], label="Parameter group {}".format(i))

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()
