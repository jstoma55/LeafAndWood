# Imports
import sys
#sys.path.append("..")
import argparse
import os
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import MinkowskiFunctional as MF

# Local Imports
from models.models import Seg1, Seg2, Seg3, count_parameters, FrankenSeg
from data.datasets import TrainingDataset, PredictionDataset
from models.util import accuracy

config_file = None
model_path = None
debug_mode = True
log_path = "logs/train_log.txt"
def log(txt):
    print(txt)
    if not debug_mode:
        with open(log_path,'a') as lf:
            lf.write(txt+"\n")
# WIP
parser = argparse.ArgumentParser(description='Train a model.')


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config = {
    "TRAIN_PATH": "h5_meter/train.h5py",
    "DEV_PATH": "h5_meter/validate.h5py",
    "BATCH_SIZE": 64,
    "NUM_CLASSES": 2,
    "N_FEATURES": 2,
    "CHANNELS":32,
    "MODEL_DEPTH":4,
    "NUM_EPOCHS": 500,
    "LEARNING_RATE": 0.001,
    "L2": 0.005,
    "MOMENTUM": 0.9,
    "QUANTIZATION_SIZE": 0.01
}
   

def train(config):
    # dataloader
    train = TrainingDataset(os.path.abspath("../data/"+config['TRAIN_PATH']), quantization_size = config['QUANTIZATION_SIZE'], debug=debug_mode)
    train_loader = torch.utils.data.DataLoader(train, batch_size = config['BATCH_SIZE'],
                                                   collate_fn=ME.utils.batch_sparse_collate)
    dev = PredictionDataset(os.path.abspath("../data/"+config['DEV_PATH']), quantization_size = config['QUANTIZATION_SIZE'])
    dev_loader = torch.utils.data.DataLoader(dev, batch_size = config['BATCH_SIZE'],
                                                   collate_fn=ME.utils.batch_sparse_collate)
    # Model, loss, optimizer
    model = Seg3(2, 32, 4, 2).to(device)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr = config['LEARNING_RATE'], weight_decay = config['L2'])
    count_parameters(model)

    # Training 
    log("Training...")
    for epoch in range(1,config['NUM_EPOCHS']+1):
        log("Starting epoch "+str(epoch))
        total_loss = 0
        total=0
        correct=0
        model.train()
        for i, (coords, feats, labels) in enumerate(train_loader):
            optimizer.zero_grad()            
            y_ = model(ME.SparseTensor(feats.float(), coords, device=device))
            y__ = MF.softmax(y_, dim=1).F.squeeze()
            loss = loss_func(y__, labels.long().to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            c, t = accuracy(y__, labels)
            correct += c
            total += t
            if i % 100 == 0:
                print(f"Cumulative train accuracy at batch {i}: {1.0*correct/total}")
            del coords, feats, labels,y_, y__,loss
            torch.cuda.empty_cache()
            #torch.cuda.synchronize()
        correct = 0
        total = 0
        dev_loss = 0
        model.eval()
        for i, (coords, feats, labels) in enumerate(dev_loader):
            in_field = ME.TensorField(
                features=feats.float(),
                coordinates=coords,
                quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE, device=device)
            x = in_field.sparse()
            y_ = model(x)
            y__ = MF.softmax(y_.slice(in_field), dim=1).F.squeeze()
            loss = loss_func(y__, labels.long().to(device))
            dev_loss += loss.item()
            c, t = accuracy(y__, labels)
            correct += c
            total += t
            del coords, feats, labels,y_, y__, loss
            torch.cuda.empty_cache()
            #torch.cuda.synchronize()
        log(f"Epoch {epoch} complete: train loss: {total_loss}, dev loss: {dev_loss}, dev accuracy: {1.0*correct/total}")
        if not debug_mode and epoch % 20 == 0:
            torch.save(model.state_dict(), "../models/prototype2/snapshot_"+str(epoch)+".pt")
        

train(config)
