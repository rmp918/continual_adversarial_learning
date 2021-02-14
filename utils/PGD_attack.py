import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

from .to_var import to_var
from .clip_by_tensor import clip_by_tensor

class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.00784313725, k=7, a=1.5, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model.cpu()
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            #X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,X_nat.shape).astype('float32')
            X = X_nat + torch.FloatTensor(X_nat.shape).uniform_(-self.epsilon, self.epsilon)
            
        else:
            X = np.copy(X_nat)
        #self.model.eval()
        with torch.enable_grad():
            for i in range(self.k):
                X_var = to_var(X, requires_grad=True)
                #y_var = torch.LongTensor(y)
                #y_var = to_var(y)

                scores = self.model(X_var)
                #print(scores)
                loss = self.loss_fn(scores, y)
                loss.backward()
                grad = X_var.grad.data.cpu()

                X += self.a * torch.sign(grad)

                max_x = X_nat + self.epsilon
                min_x= X_nat - self.epsilon
                
                #X = torch.max(torch.min(X, max_x), min_x)
                X = clip_by_tensor(X, min_x, max_x)
                #X = torch.clamp(X, X_nat - self.epsilon, X_nat + self.epsilon)
                X = torch.clamp(X, 0, 1) # ensure valid pixel range
            
        
        #self.model.train()
        return X
