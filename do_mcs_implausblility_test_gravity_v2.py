import glob
from typing import List

import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random

from shapely.geometry import Polygon
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.004, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/nonstochastic', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=1000, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--use_edge_kernels', action='store_true')
parser.add_argument('--dataset', default='mcs_test', help='dataset to train with')
parser.add_argument('--mcs_task', default='GravitySupportEvaluation', help='mcs task')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=55, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128,
                    help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=8, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true',
                    help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
opt = parser.parse_args()
BATCH_SIZE = opt.batch_size
saved_model = None
if opt.model_dir != '':
    models = glob.glob(f'{opt.model_dir}/model_*.pth')
    latest_model = sorted(models, key=lambda s: int(s[s.rfind('_e') + 2: s.rfind('.pth')]), reverse=True)[0]
    print('Loading model ', latest_model)
    saved_model = torch.load(latest_model)
    niter = opt.niter
    dataset = opt.dataset
    mcs_task = opt.mcs_task
    n_future = opt.n_future
    opt = saved_model['opt']
    opt.batch_size = BATCH_SIZE
    opt.niter = niter  # update number of epochs to train for
    opt.dataset = dataset
    opt.mcs_task = mcs_task
    opt.n_future = n_future
else:
    raise ValueError("Please specify the model to load with the --model_dir argument")

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30, 30)
fontScale = 0.6
fontColor = (0, 0, 0)
thickness = 1

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

import models.lstm as lstm_models

if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    frame_predictor.batch_size = BATCH_SIZE
    posterior = saved_model['posterior']
    posterior.batch_size = BATCH_SIZE
    prior = saved_model['prior']
    prior.batch_size = BATCH_SIZE
else:
    raise ValueError('Please specify --model_dir')

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model
    elif opt.image_width == 128:
        import models.dcgan_128 as model
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)

if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    raise ValueError("Please specify the model to load with the --model_dir argument")

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()


def kl_criterion(mu, logvar):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= opt.batch_size
    return KLD


# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

print(opt)

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt, sequential=True, implausible=False)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=False,
                          drop_last=True,
                          pin_memory=True, )
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)


def get_training_batch():
    while True:
        for sequence, labels in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch, labels


training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence, labels in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch, labels


testing_batch_generator = get_testing_batch()


def get_center_maximal_contour(img, draw=False):
    if len(img.shape) == 3:  # bgr 2 gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) == 0:
        # print(max_contour)
        # print(tuple(max_contour[0][0]))
        if draw:
            cv2.circle(img, tuple(max_contour[0][0]), 1, 150, 1)
        return tuple(max_contour[0][0])
    M = cv2.moments(max_contour)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    if draw:
        cv2.circle(img, (cx, cy), 1, 150, 1)
    return (cx, cy)


def get_polygons_from_img(img, top_n_contours=2):
    img = high_pass(img.copy(), min=9, rad=7)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:top_n_contours]
    img = cv2.drawContours(img, contours, -1, 180, 1)
    # contours = filter(lambda x: cv2.contourArea(x) > 0, contours)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) == 0:
            continue
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        poly_approx = cv2.approxPolyDP(cnt, epsilon, True)  # shape: NUM_POINTS, 1, 2 (x and y locations)
        poly_approx = np.squeeze(poly_approx, axis=1)  # compress the second dimension -> (NUM_POINTS, 2)
        try:
            poly_approx = Polygon(poly_approx)
        except ValueError:
            print('too few vertices for polygon. original list: ', poly_approx)
            idx = np.random.choice(list(range(len(poly_approx))), 3, replace=True)
            poly_approx = Polygon(poly_approx[idx])
            print('resampled list: ', poly_approx)
        # poly_approx -= get_center_of_mass(poly_approx)  # center the COM at origin
        c = poly_approx.centroid.coords[0]
        cv2.circle(img, (round(c[0]), round(c[1])), 1, 150, 1)
        polys.append(poly_approx)
    return polys, img


def get_implausibility_score(poly_list1: List[Polygon], poly_list2: List[Polygon], thresh=5):
    # thresh: when max of min distance > thresh, implausibility score > 0.5
    min_dists = []
    for p1 in poly_list1:
        p1_centeroid = p1.centroid
        min_dist = 100000000000
        for p2 in poly_list2:
            min_dist = min(min_dist, p1_centeroid.distance(p2.centroid))
        min_dists.append(min_dist)

    poly_list1, poly_list2 = poly_list2, poly_list1
    for p1 in poly_list1:
        p1_centeroid = p1.centroid
        min_dist = 100000000000
        for p2 in poly_list2:
            min_dist = min(min_dist, p1_centeroid.distance(p2.centroid))
        min_dists.append(min_dist)
    if len(min_dists) == 0:
        return 0  # if both the source and prediction has nothing going on, we say it's plausible
    max_min_dist = max(min_dists)
    # plausibility = 1 / (max(3, max_min_dist) - 2)  # R+ -> [0,1]
    # return 1 - plausibility
    x = (max_min_dist - thresh) / 2  # this means that when MMD >= 4, we have implausibility score >= 0.5 and <0.5 if not.
    implausibility = 1 / (1 + np.exp(-x))
    return implausibility


def high_pass(frame, min=10, rad=5):
    fr = frame.copy()
    fr[fr < min] = 0  # remove low brightness pixels
    if np.sum(fr) != 0:
        frame = fr

    frame = cv2.medianBlur(frame, rad)
    # frame = get_maximal_contour(frame)
    return frame


def do_implasubility_test(z_residual_mean, z_residual_cov, thresh, visualize=True):
    MOTION_THRESH = 0.001
    epoch_size = 200 // opt.batch_size  # we have a total of 1000 videos
    cov_inv = [np.linalg.pinv(cov, hermitian=True) for cov in
               z_residual_cov]  # we assume the covariance matrix is diagonal and do the inverse for each time
    # cov_inv = [np.linalg.inv(np.diag(np.diag(cov))) for cov in z_residual_cov]
    # cov_inv = [np.eye(32) for cov in z_residual_cov]
    frame_predictor.eval()
    posterior.eval()
    encoder.eval()
    decoder.eval()
    progress = progressbar.ProgressBar(max_value=epoch_size).start()
    confusion_matrix = [[0, 0], [0, 0]]
    # for i in range(50):
    #     if i % 2 == 0:
    #         x = next(training_batch_generator)
    #     else:
    #         x = next(training_batch_generator_implausible)
    for i in range(epoch_size):
        h_residual_var = torch.tensor(np.zeros(opt.n_future, dtype=np.float32), requires_grad=False,
                                      device=torch.device('cpu'))
        scores = np.array([0 for i in range(opt.n_future)], dtype=np.float32)
        progress.update(i + 1)
        is_implausible = False
        try:
            x, labels = next(training_batch_generator)
            x_diff = [torch.abs(x[t] - x[t - 1]).detach() for t in range(1, len(x))]
            x_diff = [torch.mean(frame, dim=(1, 2, 3)) for frame in x_diff]  # mean along C, H, W
            motion_start_time = []

            # find when motion starts (object is about to be dropped)
            for batch in range(len(x_diff[0])):
                started = False
                for t in range(30, len(x_diff)):
                    if x_diff[t][batch] > MOTION_THRESH:
                        motion_start_time.append(
                            t + 1 + 1)  # add one because x_diff starts at t=1, then add another frame
                        started = True
                        break
                if not started:
                    motion_start_time.append(33)  # default to frame 33 if no motion is found
            frames = [frame.cpu() for frame in x]
            # print(frames[0][0].numpy().dtype)
            # print(frames[0][0])
            # quit()
        except TypeError:
            print('got None at i = {}, terminating'.format(i))
            break
        h_posterior = [encoder(x[j]) for j in range(0, opt.n_past + opt.n_future)]
        # print(h_posterior[0][0].size())
        last_pred = None
        frame_predictor.hidden = frame_predictor.init_hidden()
        prior.hidden = prior.init_hidden()
        posterior.hidden = posterior.init_hidden()

        # start is the first frame that the prior model sees (so start + n_past is the first frame predicted)
        if motion_start_time[0] is not None:
            start = motion_start_time[0] - 4
            x_in = x[start]
            x_out_seq = [x[start].cpu()]
            for j in range(start + 1, opt.n_past + opt.n_future + 30):
                h = encoder(x_in)
                if opt.last_frame_skip or j < opt.n_past + start - 1:
                    h, skip = h
                else:
                    h, _ = h

                if j < opt.n_past + start + 5:
                    z_t = posterior(h_posterior[j][0].detach())
                    prior(h)
                    h_post = frame_predictor(torch.cat([h, z_t], 1))
                    x_in = x[j]
                    x_out_seq.append(decoder([h_post, skip]).detach().cpu())
                else:
                    z_t_hat = prior(h)
                    h = frame_predictor(torch.cat([h, z_t_hat], 1)).detach()
                    x_in = decoder([h, skip])
                    x_out_seq.append(x_in.detach().cpu())

                if j >= opt.n_past + opt.n_future and torch.mean(torch.abs(x_out_seq[-1] - x_out_seq[-2]),
                                                                 dim=(1, 2, 3)) <= MOTION_THRESH / 3:
                    # print('broke at j= ', j)
                    break

        # z_residual_scores_filtered = -0.25 * scores[:-2] + (0.5 + 0.5) * scores[1:-1] - 0.25 * scores[2:]
        # print(motion_start_time[0], len(frames), len(x_out_seq))
        # print(h_residual_var)
        background = frames[0][0][0].numpy().copy()
        background = np.uint8(np.minimum(background / 1.2, 1.0) * 255.)
        if visualize and motion_start_time[0] is not None:
            k = 0
            j = start
            source_center = None
            pred_center = None
            while not (j >= len(frames) and k >= len(x_out_seq)):
                frame_cv2 = frames[min(j, len(frames) - 1)][0][0].numpy().copy()
                frame_cv2 /= 1.2
                frame_cv2 = np.uint8(np.minimum(frame_cv2, 1.0) * 255.)
                source_diff = cv2.absdiff(frame_cv2, background)
                polys_source, source_diff_marked = get_polygons_from_img(source_diff)
                # source_center = get_center_maximal_contour(source_diff, draw=True)
                cv2.imshow('source', cv2.resize(frame_cv2, (384, 384), interpolation=cv2.INTER_NEAREST))
                cv2.imshow('source diff', cv2.resize(source_diff_marked, (384, 384), interpolation=cv2.INTER_NEAREST))

                out_cv2 = x_out_seq[min(k, len(x_out_seq) - 1)][0][0].numpy().copy()
                out_cv2 /= 1.2
                # out_cv2 /= np.max(out_cv2)
                out_cv2 = np.uint8(np.minimum(out_cv2, 1.0) * 255.)
                pred_diff = cv2.absdiff(out_cv2, background)
                polys_pred, pred_diff_marked = get_polygons_from_img(pred_diff)
                # pred_center = get_center_maximal_contour(pred_diff, draw=True)
                cv2.imshow('prediction from first 5 frames',
                           cv2.resize(out_cv2, (384, 384), interpolation=cv2.INTER_NEAREST))
                cv2.imshow('pred diff',
                           cv2.resize(pred_diff_marked, (384, 384), interpolation=cv2.INTER_NEAREST))
                src_pred_diff = cv2.absdiff(source_diff, pred_diff)
                imp_score = get_implausibility_score(polys_source, polys_pred)
                src_pred_diff = cv2.resize(src_pred_diff, (384, 384), interpolation=cv2.INTER_NEAREST)
                cv2.putText(src_pred_diff, str(imp_score), (20, 20), font, fontScale, 128, thickness)
                cv2.imshow('source-pred diff', src_pred_diff)
                cv2.waitKey(0)
                j += 1
                k += 1
        if not visualize:
            frame_cv2 = frames[-1][0][0].numpy().copy()
            frame_cv2 = np.uint8(np.minimum(frame_cv2 / 1.2, 1.0) * 255.)
            source_diff = high_pass(cv2.absdiff(frame_cv2, background))
            source_center = get_center_maximal_contour(source_diff, draw=True)
            out_cv2 = x_out_seq[-1][0][0].numpy().copy()
            out_cv2 = np.uint8(np.minimum(out_cv2 / 1.2, 1.0) * 255.)
            pred_diff = high_pass(cv2.absdiff(out_cv2, background))
            pred_center = get_center_maximal_contour(pred_diff, draw=True)

        if (source_center is not None) and (pred_center is not None):
            is_implausible = np.abs(source_center[-1] - pred_center[-1]) > thresh  # a good thresh seems to be

        if is_implausible:
            msg = 'implausible'
        else:
            msg = 'plausible'
        # print('gt: ', labels, 'prediction: ', msg)
        confusion_matrix[int(labels[0] == 'implausible')][int(is_implausible)] += 1
        if visualize:
            cv2.waitKey(0)


        # percentile = np.percentile(scores[76:152], 85.0)
        # thresh = percentile * 1 + 0.5
        # spikes_idx = np.argwhere(z_residual_scores_filtered > thresh)
        # spikes_idx = spikes_idx[
        #     (spikes_idx >= 75) & (spikes_idx <= 150)]  # ignore spikes near the start and end of video
        # spikes = z_residual_scores_filtered[spikes_idx]
        # msg = ''
        # if len(spikes_idx) > 0:
        #     # we add n_past because the first n_past frames are not counted. Add 1 because of the filtering
        #     msg = 'thresh {:.1f} IMPLAUSIBLE spikes: '.format(thresh) + str(
        #         ['{:.2f}@{}'.format(z_residual_scores_filtered[k], k + opt.n_past + 1) for k in spikes_idx])
        #     confusion_matrix[is_implausible][1] += 1
        # else:
        #     max_idx = np.argmax(z_residual_scores_filtered[75:151]) + 75
        #     msg = 'thresh {:.1f} PLAUSIBLE max {:.2f}@{}'.format(thresh, z_residual_scores_filtered[max_idx],
        #                                                          max_idx + opt.n_past + 1)
        #     confusion_matrix[is_implausible][0] += 1
        #
        # print(msg)
        # plt.savefig('implausibility_test.png')
    # h_residual_var /= epoch_size  # get the mean error vector per time
    # h_residual_sd = torch.sqrt(h_residual_var)
    print('Last i = {}'.format(i))

    # H_err_cov = torch.tensor(np.zeros((opt.n_future, 128, 128), dtype=np.float32), requires_grad=False,
    #                          device=torch.device('cuda:0'))

    # plot some stuff
    return confusion_matrix


# f = open('new_mcs_stats_post.json', 'r')
# mcs_stats_dict = json.load(f)
mcs_stats_dict = {}
with open('new_mcs_stats_post.npy', 'rb') as f:
    mcs_stats_dict['mean'] = np.load(f)
    mcs_stats_dict['cov'] = np.load(f)

ROC_curve = {}
for thr in range(2, 11):
    train_data, test_data = utils.load_dataset(opt, sequential=True, implausible=False)

    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              drop_last=True,
                              pin_memory=True, )
    training_batch_generator = get_training_batch()

    conf_mat = do_implasubility_test(np.array(mcs_stats_dict['mean']), np.array(mcs_stats_dict['cov']), thr, visualize=True)
    ROC_curve[thr] = conf_mat
print(ROC_curve)
