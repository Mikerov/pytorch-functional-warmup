import types
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class FuncLRScheduler(_LRScheduler):

    def __init__(self, optimizer: Optimizer, lr_goal, warm_epochs, last_epoch: int = -1, fns=None,
                 scheduler_after: _LRScheduler = None):
        self.optimizer = optimizer
        self.func = []
        self.warm_epochs = []
        self.last_epoch = last_epoch
        self.last_epochs = [0] * len(warm_epochs)

        if not isinstance(fns, list) and not isinstance(fns, tuple):
            self.func = [fns] * len(optimizer.param_groups)
        for item in fns:
            if isinstance(item, types.FunctionType):
                self.func.append(item)
            else:
                raise ValueError("Expected fn to be a function, but got {}".format(type(fns)))
        if len(fns) != len(optimizer.param_groups):
            raise ValueError("Expected {} functions, but got {}".format(
                len(optimizer.param_groups), len(fns)))

        if not isinstance(warm_epochs, list) and not isinstance(warm_epochs, tuple):
            self.warm_epochs = [warm_epochs] * len(optimizer.param_groups)
        for item in warm_epochs:
            if isinstance(item, int):
                self.warm_epochs.append(item)
            else:
                raise ValueError("Expected warm_epochs to be an int, but got {}".format(type(fns)))
        if len(warm_epochs) != len(optimizer.param_groups):
            raise ValueError("Expected {} warm_epochs, but got {}".format(
                len(optimizer.param_groups), len(warm_epochs)))

        if not isinstance(lr_goal, list) and not isinstance(lr_goal, tuple):
            self.lr_goal = [lr_goal] * len(optimizer.param_groups)
        else:
            if len(lr_goal) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_goal)))
            self.lr_goal = list(lr_goal)

        if isinstance(scheduler_after, _LRScheduler):
            self.next_scheduler = scheduler_after
        else:
            raise ValueError("Expected scheduler_after to be a scheduler, but got {}".format(type(scheduler_after)))

        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_goal)

        for idx, fn in enumerate(self.lr_goal):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_goal[idx].__dict__.update(fn)

    def get_lr(self):
        to_return = [func(self.last_epoch, lr_goal, warm_epochs) for func, lr_goal, warm_epochs in
                     zip(self.func, self.lr_goal, self.warm_epochs)]
        for index, item in enumerate(to_return):
            if item < 0:
                remember = self.next_scheduler.last_epoch
                self.next_scheduler.last_epoch = self.last_epochs[index] + 1
                to_return[index] = self.next_scheduler.get_lr()[index]
                self.last_epochs[index] = self.next_scheduler.last_epoch
                self.next_scheduler.last_epoch = remember
        return to_return
