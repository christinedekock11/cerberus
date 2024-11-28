from scipy import sparse, io
import glob
import numpy as np
from random import shuffle
import torch

class Reader(object):
    def __init__(self, val_frac=0.1):
        self.train_inds = []
        self.val_inds = []
        self.mats = []
        self.val_frac = val_frac
        self.val_frac=val_frac

    def create_tuples(self, fname, timestep):
        mat = io.mmread(fname)
        empty = np.ravel(mat.sum(axis=1) == 0)
        tuples = np.arange(mat.shape[0])[~empty]
        tuples = list(zip([timestep]*len(tuples),tuples))

        shuffle(tuples)
        nval = round(self.val_frac*len(tuples))
        self.train_inds.extend(tuples[:-nval])
        self.val_inds.extend(tuples[-nval:])
        self.mats.append(mat)

    def run(self, path):
        files = list(sorted(glob.glob(path)))
        _ = [self.create_tuples(f, t) for t, f in enumerate(files)]
        return self.mats, self.train_inds, self.val_inds

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, mats, inds, source):
        self.inds = inds
        self.mats = mats
        self.source = source

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        t, u = self.inds[idx]
        target = self.mats[t].getrow(u).toarray()[0]
        return t, u, target, self.source


