import logging
import random
import os

import cv2
import numpy as np
from glob import glob
import torch
import scipy.misc
import imageio
from os import path


class MCS(object):

    def __init__(self, train, data_root, seq_len=20, image_size=64, task='ALL', sequential=None, implausible=False,
                 test_set=False, im_channels=1, use_edge_kernels=True, labels=False, start_min=None, start_max=None, sequence_stride=None,
                 reduce_static_frames=False):
        # if implausible is set to True, generates "fake" images by cutting out or repeating frames
        self.implausible = implausible
        if test_set:
            self.data_root = '%s/mcs_videos_test/processed/' % data_root
        else:
            self.data_root = '%s/mcs_videos_1000/processed/' % data_root
        if not os.path.exists(self.data_root):
            raise os.error('data/mcs.py: Data directory not found!')
        self.seq_len = seq_len
        self.image_size = image_size
        self.im_channels = im_channels
        self.labels = labels
        if use_edge_kernels:
            if im_channels != 1:
                raise AssertionError('Using edge kernels implies the output images are grayscale! Set im_channels to 1!')
        self.use_edge_kernels = use_edge_kernels
        self.start_min = start_min
        self.start_max = start_max
        self.sequence_stride = sequence_stride
        self.reduce_static_frames = reduce_static_frames
        self.motion_threshold = 0.001

        # print('mcs.py: found tasks ', self.tasks)
        self.video_folder = {}
        self.len_video_folder = {}
        if task == 'ALL':
            self.tasks = [os.path.basename(folder) for folder in sorted(glob(path.join(self.data_root, '*')))]
        else:
            self.tasks = [task]

        for task in self.tasks:
            self.video_folder[task] = [path.basename(folder) for folder in
                                       sorted(glob(path.join(self.data_root, task, '*')))]
            self.len_video_folder[task] = len(self.video_folder[task])
        self.seed_set = False
        self.sequential = sequential  # if set to true, return videos in sequence

    def get_sequence(self, idx=None):
        stride = max(1, self.sequence_stride)
        if not self.sequential:
            task = random.choice(self.tasks)
            vid = random.choice(self.video_folder[task])
            num_frames = len(next(os.walk(path.join(self.data_root, task, vid)))[2])
            frame_path = path.join(self.data_root, task, vid, vid + '_')
        else:
            assert len(self.tasks) == 1 and idx is not None
            # index = idx % self.len_video_folder[task] # loop over the videos under a task
            task = self.tasks[0]
            if idx >= self.len_video_folder[task]:
                return None  # we've run out of videos
            vid = self.video_folder[task][idx]
            num_frames = len(next(os.walk(path.join(self.data_root, task, vid)))[2])  # how many frames we can use after considering the stride
            frame_path = path.join(self.data_root, task, vid, vid + '_')
        label = str(os.path.basename(vid))
        label = label[label.rfind('_') + 1:]

        start_min = 0
        start_max = num_frames - 1 - (self.seq_len - 1) * stride
        if start_max < 0:
            raise ValueError("Number of frames in the dataset less than the desired sequence length!")
        if self.start_min:
            start_min = self.start_min
        if self.start_max:
            assert self.start_max <= start_max  # so the sequence doesn't start too late
            start_max = self.start_max
        assert start_min <= start_max
        start = random.randint(start_min, start_max)
        seq = []
        last_im = None
        first_movement = None
        first_static = None
        second_movement = None
        # choose a random subsequence of frames in the selected video
        if not self.reduce_static_frames:
            end = start + self.seq_len * stride
        else:
            end = num_frames
        for i in range(start, end, stride):
            # i is 0-indexed so we need to add 1 to i
            fname = frame_path + f'{i + 1:04d}.png'
            im = imageio.imread(fname) / np.float32(255.)
            if self.im_channels == 1:  # convert to grayscale if specified
                if not self.use_edge_kernels:
                    # regular grayscale conversion
                    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
                    im = gray(im)[..., np.newaxis]
                else:
                    edge_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                    ddepth = cv2.CV_32F
                    scale = 1
                    delta = 0
                    for channel in range(3):
                        color = im[..., channel]
                        # grad_x = cv2.Sobel(color, ddepth, 1, 0, ksize=5, scale=scale, delta=delta,
                        #                    borderType=cv2.BORDER_DEFAULT)
                        # grad_y = cv2.Sobel(color, ddepth, 0, 1, ksize=5, scale=scale, delta=delta,
                        #                    borderType=cv2.BORDER_DEFAULT)
                        grad_x = cv2.Scharr(color, ddepth, 1, 0, scale=scale, delta=delta,
                                            borderType=cv2.BORDER_DEFAULT)
                        grad_y = cv2.Scharr(color, ddepth, 0, 1, scale=scale, delta=delta,
                                            borderType=cv2.BORDER_DEFAULT)
                        # abs_grad_x = np.abs(grad_x)
                        # abs_grad_y = np.abs(grad_y)
                        edge_map += np.sqrt(grad_x**2 + grad_y**2)
                    edge_map /= 3  # 3 channels
                    edge_map /= 12  # to reduce magnitude
                    im = edge_map[..., np.newaxis]
            if self.reduce_static_frames and last_im is not None:
                motion_magnitude = np.mean(cv2.absdiff(im, last_im)) * 256
                print(motion_magnitude)
                if first_movement is None and motion_magnitude > self.motion_threshold:
                    first_movement = i
                elif first_movement is not None and motion_magnitude <= self.motion_threshold:
                    first_static = i
                elif first_static is not None and motion_magnitude > self.motion_threshold:
                    second_movement = i
            last_im = im
            seq.append(im)
        if self.reduce_static_frames:
            new_seq = []

            # keep the first and last static frames, then add the frames before and after to fill the sequence
            len_before = (self.seq_len - 2) // 2
            len_after = (self.seq_len - 2) - len_before
            new_seq += seq[first_static - start - len_before: first_static - start + 1]
            new_seq += seq[second_movement - 1 - start: second_movement - start + len_after]

            print(first_static, second_movement)
            print(first_static - start - len_before, first_static - start + 1)
            print(second_movement - 1 - start, second_movement - start + len_after)
            print(len(new_seq), len(seq))
            for img in new_seq:
                cv2.imshow('img', img[:, :, 0])
                cv2.waitKey(0)
        if self.labels:
            return np.array(seq), label
        else:
            return np.array(seq)

    def abnormalize_sequence(self, seq):
        """
        Takes a sequence and makes it implausible by cutting out and then repeating frames or
        repeating frames and then cutting out frames
        """
        implausibility_type = random.randint(1, 3)
        # start = random.randint(100, 140)
        start = 115
        vid_len = len(seq)
        duration = 7
        implausibility_type = 1
        if implausibility_type == 1:  # object is invisible when/where it shouldn't be
            no_object_frame = seq[30]
            seq[start:start + duration] = no_object_frame
        elif implausibility_type == 2:  # object suddenly freezes then teleports where it would be
            seq[start:start + duration] = seq[start]
        elif implausibility_type == 3:  # object jumps forward 10 frames and keeps moving
            seq[start:vid_len - duration] = \
                seq[start + duration: vid_len]
        elif implausibility_type == 0:
            pass  # do nothing if type if 0
        return seq

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            # torch.manual_seed(index)
        if self.labels:
            seq, labels = self.get_sequence(index)
        else:
            seq = self.get_sequence(index)

        if seq is not None:
            if self.implausible:
                seq = self.abnormalize_sequence(seq)
            if self.labels:
                return torch.from_numpy(seq), labels
            else:
                return torch.from_numpy(seq)
        else:
            return None

    def __len__(self):
        return 5 * 1000 * 200  # approximate

# if __name__ == '__main__':
#     m = MCS(True, '/home/lol/Hub/svg/data/')
#     s = m.__getitem__(0).cuda()
#     print(s.device)
