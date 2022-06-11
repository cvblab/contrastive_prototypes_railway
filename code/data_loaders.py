import torch
import numpy as np
import random
import skimage
import os
from PIL import Image
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_dataset, items, input_shape):
        self.dir_dataset = dir_dataset
        self.items = items
        self.input_shape = input_shape
        self.files = os.listdir(self.dir_dataset + self.items[0])
        self.labels = 0

        # Remove other files
        self.files = [self.files[i] for i in range(len(self.files)) if self.files[i] != 'Thumbs.db']

        self.X = np.zeros((len(self.files), input_shape[0], input_shape[1], input_shape[2]))

        print('[INFO]: Training on ram: Loading images')
        for iFile in np.arange(0, len(self.files)):
            print(str(iFile) + '/' + str(len(self.files)), end='\r')
            for item in np.arange(0, len(self.items)):
                x = Image.open(os.path.join(self.dir_dataset + self.items[item], self.files[iFile]))
                x = np.asarray(x) / 255.
                x = cv2.resize(x, (self.input_shape[2], self.input_shape[1]))
                self.X[iFile, item, :, :] = x

        self.indexes = np.arange(0, self.X.shape[0])
        self.channel_first = True

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        image = self.X[idx, :, :, :].copy()

        return image

    def label_cruces_adif(self):

        self.labels = np.zeros(len(self.files))
        for iFile in np.arange(0, len(self.files)):
            i_label = int(self.files[iFile][-5])
            if i_label >= 1:
                i_label = 1
            self.labels[iFile] = i_label


class Generator(object):
    def __init__(self, train_dataset, bs, shuffle=True):
        self.dataset = train_dataset
        self.bs = bs
        self.shuffle = shuffle
        self.indexes = train_dataset.indexes.copy()
        self._idx = 0

    def __len__(self):
        return round(len(self.indexes) / self.bs)

    def __iter__(self):

        return self

    def __next__(self):

        if self._idx + self.bs >= len(self.indexes):
            self._reset()
            raise StopIteration()

        # Load images and include into the batch
        X = []
        for i in np.arange(self._idx, self._idx + self.bs):

            x = self.dataset.__getitem__(self.indexes[i])
            X.append(x)

        self._idx += self.bs

        return np.array(X)

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


class GeneratorFSL(object):
    def __init__(self, train_dataset, n=4, m=4, shuffle=True, classes=[0, 1]):
        self.dataset = train_dataset
        self.n = n  # queries
        self.m = m  # support
        self.shuffle = shuffle
        self.indexes = train_dataset.indexes.copy()
        self._idx = 0
        self.classes = classes
        self.Y = self.dataset.labels

    def __len__(self):
        return round(len(self.indexes) / self.n)

    def __iter__(self):

        return self

    def __next__(self):

        if self._idx + self.n >= len(self.indexes):
            self._reset()
            raise StopIteration()

        # Load query samples
        Xq = []
        Y = []
        for i in np.arange(self._idx, self._idx + self.n):
            x = self.dataset.__getitem__(self.indexes[i])
            Xq.append(x)
            y_i = np.zeros((int(len(self.classes))))
            y_i[int(self.Y[self.indexes[i]])] = 1.
            Y.append(y_i)

        # Load query samples
        Xs = []
        for iClass in self.classes:
            Xs_i = []
            queries = np.random.choice(np.squeeze(np.array(self.indexes)[np.argwhere(self.Y[self.indexes] == iClass).astype(int)]), self.m)
            for i in queries:
                x = self.dataset.__getitem__(i)
                Xs_i.append(x)
            Xs.append(Xs_i)

        self._idx += self.n

        return np.array(Xs), np.array(Xq), Y

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0