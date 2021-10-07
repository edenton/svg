import glob

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
BATCH_SIZE = opt.batch_size
saved_model = None
if opt.model_dir != '':
    models = glob.glob(f'{opt.model_dir}/model_*.pth')
    latest_model = sorted(models, key=lambda s: int(s[s.rfind('_e') + 2: s.rfind('.pth')]), reverse=True)[0]
    print('Loading model ', latest_model)
    saved_model = torch.load(latest_model)
    model_dir = opt.model_dir
    niter = opt.niter
    opt = saved_model['opt']
    opt.niter = niter  # update number of epochs to train for
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    raise ValueError("Please specify the model to load with the --model_dir argument")

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
    raise ValueError("Please specify the model to load with the --model_dir argument")

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
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

opt.batch_size = BATCH_SIZE
opt.epoch_size = 1000
opt.n_future = 195
print(opt)

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt, sequential=True)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          drop_last=True,
                          pin_memory=True)
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


training_batch_generator = get_training_batch()
training_batch_generator_2 = get_training_batch()


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


def do_stats():
    epoch_size = 1000 // opt.batch_size // 1  # we have a total of 1000 videos

    frame_predictor.eval()
    prior.eval()
    posterior.eval()
    encoder.eval()
    decoder.eval()
    progress = progressbar.ProgressBar(max_value=epoch_size).start()
    z_residual_mean = torch.tensor(np.zeros((opt.n_future, opt.z_dim), dtype=np.float64), requires_grad=False,
                                   device=torch.device('cuda:0'))
    i = 0
    for i in range(epoch_size):
        progress.update(i + 1)
        try:
            x = next(training_batch_generator)
        except TypeError:
            print('got None at i = {}, terminating'.format(i))
            break
        # print(h_posterior[0][0].size())
        frame_predictor.hidden = frame_predictor.init_hidden()
        prior.hidden = prior.init_hidden()
        posterior.hidden = posterior.init_hidden()

        last_h = encoder(x[0])
        for j in range(1, opt.n_past + opt.n_future):
            h_target = encoder(x[j])
            if opt.last_frame_skip or j < opt.n_past:
                h, skip = last_h
                h = h.detach()
            else:
                h = last_h[0].detach()
            # we predict h_t from h_{t-1}
            z_t = posterior(h_target[0].detach()).detach()
            z_t_hat = prior(h).detach()

            if j >= opt.n_past:
                # h_res = h_prior_pred - h_posterior[j][0].detach()  # predicted h minus observed h
                # h_res = h_prior_pred
                # h_res = torch.mean(h_res, dim=0)  # average errors at the same time j over the batch
                # z_residual_mean[j - opt.n_past] += h_res
                residual = z_t - z_t_hat
                residual = torch.mean(residual, dim=0)
                z_residual_mean[j - opt.n_past] += residual
            last_h = h_target
    z_residual_mean /= epoch_size  # get the mean error vector per time

    # restart training dataset
    global train_loader
    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              drop_last=True,
                              pin_memory=True)

    z_cov = torch.tensor(np.zeros((opt.n_future, opt.z_dim, opt.z_dim), dtype=np.float64), requires_grad=False,
                                  device=torch.device('cuda:0'))


    for i in range(epoch_size):
        progress.update(i + 1)
        try:
            x = next(training_batch_generator_2)
        except TypeError:
            print('got None at i = {}, terminating'.format(i))
            break

        # print(h_posterior[0][0].size())
        frame_predictor.hidden = frame_predictor.init_hidden()
        prior.hidden = prior.init_hidden()
        posterior.hidden = posterior.init_hidden()
        last_h = encoder(x[0])
        for j in range(1, opt.n_past + opt.n_future):
            h_target = encoder(x[j])
            if opt.last_frame_skip or j < opt.n_past:
                h, skip = last_h
                h = h.detach()
            else:
                h = last_h[0].detach()
            # we predict h_t from h_{t-1}
            z_t = posterior(h_target[0].detach()).detach()
            z_t_hat = prior(h).detach()

            if j >= opt.n_past:
                # h_res = h_prior_pred - h_posterior[j][0].detach()  # predicted h minus observed h
                # squared_diff = torch.square(h_res - z_residual_mean[j - opt.n_past])
                # squared_diff = torch.mean(squared_diff, dim=1)  # average squared residuals at time j over the dimensions of h
                # squared_diff = torch.mean(squared_diff, dim=0)  # average squared residuals at time j over the batch
                # h_residual_var[j - opt.n_past] += squared_diff

                residual = z_t - z_t_hat  # B x D
                # B x D x 1 * B x 1 x D -> B x D x D
                sample_cov = torch.matmul(residual[:, :, np.newaxis], residual[:, np.newaxis, :])
                sample_cov = torch.mean(sample_cov, axis=0)  # mean over batch dimension
                z_cov[j - opt.n_past] += sample_cov
            last_h = h_target
    z_cov /= epoch_size
    z_cov = z_cov.cpu().numpy()
    z_sd = [np.sqrt(np.diag(cov)) for cov in z_cov]
    z_sd = np.array(z_sd)
    print('Last i = {}'.format(i))
    print('sd of z residual: ', z_sd)
    print('norm(mean of z residual)', torch.norm(z_residual_mean, dim=1))

    # H_err_cov = torch.tensor(np.zeros((opt.n_future, 128, 128), dtype=np.float32), requires_grad=False,
    #                          device=torch.device('cuda:0'))

    # plot some stuff
    z_residual_mean_norm = torch.norm(z_residual_mean, dim=1).cpu()
    plt.subplot(3, 1, 1)
    plt.xlabel('Time')
    plt.tight_layout()
    plt.title("Norm of the residual mean")
    plt.bar(np.arange(len(z_residual_mean_norm)), z_residual_mean_norm)

    plt.subplot(3, 1, 2)
    plt.xlabel('Time')
    plt.tight_layout()
    plt.title("Average dimensional sqrt(variance) of the residual")
    plt.bar(np.arange(len(z_sd)), np.mean(z_sd, axis=1))
    plt.savefig('z_residual.png')

    stats_dict = {'mean': z_residual_mean.cpu().numpy(), 'cov': z_cov}
    print(stats_dict['mean'].dtype)
    print(stats_dict['cov'].dtype)
    # f = open('new_mcs_stats_post.json', 'w')
    # json.dump(stats_dict, f)
    with open('new_mcs_stats_post.npy', 'wb') as f:
        np.save(f, stats_dict['mean'])
        np.save(f, stats_dict['cov'])


do_stats()
