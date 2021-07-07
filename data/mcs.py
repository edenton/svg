import logging
import random
import os
import numpy as np
from glob import glob
import torch
import scipy.misc
import imageio
from os import path

class MCS(object):

    def __init__(self, train, data_root, seq_len = 20, image_size=64, task='ALL'):
        self.data_root = '%s/mcs_videos_1000/processed/' % data_root
        if not os.path.exists(self.data_root):
            raise os.error('data/mcs.py: Data directory not found!')
        self.seq_len = seq_len
        self.image_size = image_size 

        # print('mcs.py: found tasks ', self.tasks)
        self.video_folders = {}
        if task == 'ALL':
            self.tasks = [os.path.basename(folder) for folder in glob(path.join(self.data_root, '*'))]
        else:
            self.tasks = [task]

        for task in self.tasks:
            self.video_folders[task] = [path.basename(folder) for folder in glob(path.join(self.data_root, task, '*'))]

        self.seed_set = False

    def get_sequence(self):
        task = random.choice(self.tasks)
        vid = random.choice(self.video_folders[task])
        num_frames = len(next(os.walk(path.join(self.data_root, task, vid)))[2])  # dir is your directory path as string

        frame_path = path.join(self.data_root, task, vid, vid + '_')

        start = random.randint(0, num_frames-self.seq_len)
        seq = []
        for i in range(start, start+self.seq_len):
            # i is 0-indexed so we need to add 1 to i
            fname = frame_path + f'{i + 1:04d}.png'
            im = imageio.imread(fname)/255.
            gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            im = gray(im)[..., np.newaxis]
            seq.append(im)
        return np.array(seq)

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            #torch.manual_seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return 5*1000*200  # approximate


# if __name__ == '__main__':
#     m = MCS(True, '/home/lol/Hub/svg/data/')
#     s = m.__getitem__(0).cuda()
#     print(s.device)

