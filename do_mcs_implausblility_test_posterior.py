import glob

import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
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
parser.add_argument('--dataset', default='mcs', help='dataset to train with')
parser.add_argument('--mcs_task', default='SpatioTemporalContinuityTraining4', help='mcs task')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=195, help='number of frames to predict')
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
BATCH_SIZE = 1
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
    data_root = opt.data_root
    opt = saved_model['opt']
    opt.batch_size = BATCH_SIZE
    opt.niter = niter  # update number of epochs to train for
    opt.dataset = dataset
    opt.mcs_task = mcs_task
    opt.n_future = n_future
    opt.data_root = data_root
    opt.start_min = 0
    opt.start_max = None
    frame_predictor = saved_model['frame_predictor'].cuda()
    frame_predictor.batch_size = BATCH_SIZE
    posterior = saved_model['posterior'].cuda()
    posterior.batch_size = BATCH_SIZE
    prior = saved_model['prior'].cuda()
    prior.batch_size = BATCH_SIZE
    decoder = saved_model['decoder'].cuda()
    encoder = saved_model['encoder'].cuda()
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

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()


def kl_criterion(mu, logvar):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= opt.batch_size
    return KLD


# # --------- transfer to gpu ------------------------------------
# frame_predictor.cuda()
# posterior.cuda()
# encoder.cuda()
# decoder.cuda()
# mse_criterion.cuda()

opt.batch_size = BATCH_SIZE
opt.epoch_size = 1000
opt.n_future = 195
print(opt)

# --------- load a dataset ------------------------------------
train_data_implausible, test_data_implausible = utils.load_dataset(opt, sequential=True, implausible=True)
train_data, test_data = utils.load_dataset(opt, sequential=True, implausible=False)

train_loader_implausible = DataLoader(train_data_implausible,
                                      num_workers=opt.data_threads,
                                      batch_size=opt.batch_size,
                                      drop_last=True,
                                      pin_memory=True, )
train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          drop_last=True,
                          pin_memory=True, )
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch


def get_training_batch_implausible():
    while True:
        for sequence in train_loader_implausible:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch


training_batch_generator = get_training_batch()
training_batch_generator_implausible = get_training_batch_implausible()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch


testing_batch_generator = get_testing_batch()


# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 1
    gen_seq = [[] for _ in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    h_seq = [encoder(x[i]) for i in range(opt.n_past)]
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h_seq[i - 1]
                h = h.detach()
            elif i < opt.n_past:
                h, _ = h_seq[i - 1]
                h = h.detach()
            if i < opt.n_past:
                frame_predictor(h)
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                h = frame_predictor(h).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [[] for t in range(opt.n_eval)]
    nrow = min(opt.batch_size, 25)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        for s in range(nsample):
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for s in range(nsample):
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
    utils.save_gif(fname, gifs)


def plot_rec(x, epoch):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    h_seq = [encoder(x[i]) for i in range(opt.n_past + opt.n_future)]
    for i in range(1, opt.n_past + opt.n_future):
        h_target = h_seq[i][0].detach()
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h_seq[i - 1]
        else:
            h, _ = h_seq[i - 1]
        h = h.detach()
        if i < opt.n_past:
            frame_predictor(h)
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(h).detach()
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 25)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past + opt.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


def do_implasubility_test(z_residual_mean, z_residual_cov, visualize=True):
    epoch_size = 199 // opt.batch_size  # we have a total of 1000 videos
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
            if i % 2 == 0:
                x = next(training_batch_generator)
            else:
                x = next(training_batch_generator_implausible)
                is_implausible = True

            frames = [frame.cpu() for frame in x]
            # print(frames[0][0].numpy().dtype)
            # print(frames[0][0])
            # quit()
        except TypeError:
            print('got None at i = {}, terminating'.format(i))
            break
        h_posterior = [encoder(x[j]) for j in range(opt.n_past + opt.n_future)]
        # print(h_posterior[0][0].size())
        last_pred = None
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        start = 1
        for j in range(start, opt.n_past + opt.n_future):
            h_target = h_posterior[j][0].detach()
            if opt.last_frame_skip or j < opt.n_past + start - 1:
                h, skip = h_posterior[j - 1]
            else:
                h = h_posterior[j - 1][0].detach()
            # we predict h_t from h_{t-1}
            z_t = posterior(h_target[0].detach()).detach()
            z_t_hat = prior(h).detach()

            if j >= opt.n_past + start - 1:
                # h_res = h_prior_pred - h_posterior[j][0].detach()  # predicted h minus observed h
                # squared_diff = torch.square(h_res - h_residual_mean[j - opt.n_past])
                # squared_diff = torch.mean(squared_diff, dim=1)  # average squared residuals at time j over the dimensions of h
                # squared_diff = torch.mean(squared_diff, dim=0)  # average squared residuals at time j over the batch
                # h_residual_var[j - opt.n_past] += squared_diff

                # residual = (h_prior_pred - h_posterior_pred).cpu().detach()
                residual = (z_t - z_t_hat).cpu().detach().numpy()
                # err = (residual - z_residual_mean[j - opt.n_past])
                err = residual
                err = np.square(err)  # [batch, dim_feature]
                score = np.sum(err)
                # score = np.matmul(err[:, np.newaxis, :], cov_inv[j - opt.n_past][np.newaxis, ...])  # B*1*D * 1*D*D
                # score = np.matmul(score, err[:, :, np.newaxis])  # * B*1*D * B*D*1 -> B*1*1
                # score = numpy.nan_to_num(score)
                # score /= opt.z_dim

                scores[j - opt.n_past] = np.sqrt(score)
                # scores[j - opt.n_past] = score
                # note: scores[t]^2 ~ Chi^2_df=z_dim so E[scores[t]^2] = z_dim
                # if len(err.shape) == 2:  # [batch, dim_feature]
                #     err = err[..., np.newaxis]  # make err into a vector [batch, dim_feature, 1]
                # # print(cov_inv[j - opt.n_past].shape)
                # # print(cov_inv[j - opt.n_past])
                # # print(err.shape)
                # print(err)
                # print(np.diag(cov_inv[j - opt.n_past][0]))
                # quit()
                # mahanlanobis_dist = np.matmul(cov_inv[j - opt.n_past], err).transpose(2, 1)
                # # print(mahanlanobis_dist.shape)
                # mahanlanobis_dist = np.matmul(mahanlanobis_dist, err)  # [batch, 1, 1]
                # # print(mahanlanobis_dist.shape)
                # # print(np.sqrt(mahanlanobis_dist))
                # print(mahanlanobis_dist)
                # quit()
                # mahanlanobis_dist = torch.mean(mahanlanobis_dist)  # scalar

                # err = torch.mean(err, dim=1)  # average errs at time j over the dimensions of h
                # err = torch.mean(err, dim=0)  # average errs at time j over the batch
                # h_residual_var[j - opt.n_past] += err.detach()
                # h_residual_var[j - opt.n_past] += torch.mean(err, axis=0)

        z_residual_scores_filtered = -0.25 * scores[:-2] + (0.5+0.5) * scores[1:-1] - 0.25 *scores[2:]

        # print(h_residual_var)
        if visualize:
            for j in range(len(frames)):
                frame_cv2 = frames[j][0][0].numpy()
                # frame_cv2 /= 3
                frame_cv2 = np.uint8(np.minimum(frame_cv2, 1.0) * 255.)
                cv2.imshow('frame', frame_cv2)
                cv2.waitKey(15)

        percentile = np.percentile(scores[76:152], 85.0)
        thresh = percentile * 1 + 0.5
        spikes_idx = np.argwhere(z_residual_scores_filtered > thresh)
        spikes_idx = spikes_idx[
            (spikes_idx >= 75) & (spikes_idx <= 150)]  # ignore spikes near the start and end of video
        spikes = z_residual_scores_filtered[spikes_idx]
        msg = ''
        if len(spikes_idx) > 0:
            # we add n_past because the first n_past frames are not counted. Add 1 because of the filtering
            msg = 'thresh {:.1f} IMPLAUSIBLE spikes: '.format(thresh) + str(['{:.2f}@{}'.format(z_residual_scores_filtered[k], k + opt.n_past + 1) for k in spikes_idx])
            confusion_matrix[is_implausible][1] += 1
        else:
            max_idx = np.argmax(z_residual_scores_filtered[75:151]) + 75
            msg = 'thresh {:.1f} PLAUSIBLE max {:.2f}@{}'.format(thresh, z_residual_scores_filtered[max_idx], max_idx + opt.n_past + 1)
            confusion_matrix[is_implausible][0] += 1

        print(msg)

        if visualize:
            fig = plt.figure()
            # plt.ylim(0, 2.0)
            plt.xlabel('Time')
            plt.title("sqrt of average squared dimensional error")
            plt.bar(np.arange(len(scores)), scores)
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            plt.xlabel('Time')
            plt.title("filtered sqrt of average squared dimensional error")
            plt.bar(np.arange(len(z_residual_scores_filtered)), z_residual_scores_filtered)
            fig.canvas.draw()
            img2 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                 sep='')
            img2 = img2.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

            cv2.imshow("plot", img)
            cv2.putText(img2, msg, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, cv2.LINE_AA)
            cv2.imshow("plot2", img2)

            k = cv2.waitKey(0)
            if k == ord('q'):
                quit()
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
conf_mat = do_implasubility_test(np.array(mcs_stats_dict['mean']), np.array(mcs_stats_dict['cov']), visualize=True)
print(conf_mat)
