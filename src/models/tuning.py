import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import models.SegModel as seg

from ray import tune
from ray.tune.schedulers import ASHAScheduler

search_space = {
    "channel": tune.grid_search([31, 64, 128]),
    "kernel_size": tune.grid_search([3,4,5]),
    "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    "momentum": tune.uniform(0.1, 0.9),
}

def train(model, optimizer, train_loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.train()

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader): 
            print("Processing tile ",i)
            coords, feats, labels = data
            optimizer.zero_grad()
            y_ = model(ME.SparseTensor(feats.float(), coords, device=device))
            loss = loss_func(y_.F.squeeze(), labels.long())
            loss.backward()
            optimizer.step()
            del coords, feats, labels,y_
            torch.cuda.empty_cache()

def test(model, loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval()
    for i, data in enumerate(train_loader): 
        print("Processing tile ",i)
        coords, feats, labels = data
        y_ = model(ME.SparseTensor(feats.float(), coords, device=device))
        
        del coords, feats, labels,y_
        torch.cuda.empty_cache()