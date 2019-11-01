# Use custom function to warm up the learning rate

This implementation can be used to warm up the learning rate of each parameter group of a nn.Module with a custom function.

## Usage

The full example can be found in test.py.

Necessary steps:
1) warm up function must return a negative value if the current epoch is larger than the number of warm up epochs for this particular parameter group
2) specifiy target learning rates, functions and number of warm up epochs for each parameter group
3) assign normal scheduler after the warm up

```python
from functional_warm_up import FuncLRScheduler

def linear(last_epoch, lr_goal, warm_epochs):
    if last_epoch > warm_epochs:
        return -1
    return last_epoch * lr_goal / warm_epochs

params = [{'params': net.conv1.parameters(), 'weight_decay': 0.1},
          {'params': net.conv2.parameters(), 'weight_decay': 0.2},
          {'params': net.conv3.parameters(), 'weight_decay': 0.3}]
opt = optim.Adam(params=params)
n_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
lr_goal = [0.1, 0.3, 0.5]
fns = [linear, linear, linear]
warm_epochs = [30, 10, 70]
wu_scheduler = FuncLRScheduler(optimizer=opt, lr_goal=lr_goal, warm_epochs=warm_epochs, fns=fns,
                               scheduler_after=n_scheduler)
                               
for epoch in range(0, epochs_before_save):
    wu_scheduler.step()
```

## Result
The below image shows the learning rate warm up of three parameter groups with quadratic, linear and squared functions for 30, 10 and 70 epochs. Afterwards, the cosine annealing is used.
![Learning rate vs. epoch](Figure_1.png?raw=true "Three different warm-up scenarios")

## Saving the state dictionary

This implementation supports saving and loading of the optimizer and scheduler state dictionaries.
