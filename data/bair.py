import os
import io
from scipy.misc import imresize 
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread 


class RobotPush(object):
    
    """Data Handler that loads robot pushing data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root 
        if train:
            self.data_dir = '%s/processed_data/train' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s/processed_data/test' % self.root_dir
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000

    def get_seq(self):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]
        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq


    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()


