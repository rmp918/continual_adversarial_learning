import sys
import torch
import logging


class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        self.lrs = []

    def add(self, optimizer, lr):
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        return

    def update(self, val, num):
        self.sum += val * num
        self.n += num

    @property
    def avg(self):
        return self.sum / self.n



def classification_accuracy(output, target):#, topk=(1,)):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()
    """Computes the accuracy over the k top predictions for the specified values of k"""
    #with torch.no_grad():
    #    maxk = max(topk)
    #    batch_size = target.size(0)
#
    #    _, pred = output.topk(maxk, 1, True, True)
    #    pred = pred.t()
    #    correct = pred.eq(target.view(1, -1).expand_as(pred))
#
    #    res = []
    #    for k in topk:
    #        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #        res.append(correct_k.mul_(100.0 / batch_size))
    #    return res
    
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

def set_dataset_paths(args):
    """Set default train and test path if not provided as input."""

    if not args.train_path:
        args.train_path = 'data/%s/train' % (args.dataset)

    if not args.val_path:
        if (args.dataset in ['imagenet', 'face_verification', 'emotion', 'gender'] or
            args.dataset[:3] == 'age'):
            args.val_path = 'data/%s/val' % (args.dataset)
        else:
            args.val_path = 'data/%s/test' % (args.dataset)


def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return
