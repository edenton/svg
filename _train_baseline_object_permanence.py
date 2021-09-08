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
import json

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/nonstochastic_posterior', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=500, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int, help='number of channels for input images. ')
parser.add_argument('--use_edge_kernels', default=True, type=bool, help='whether to use edge kernels to reduce to 1 channel')
parser.add_argument('--dataset', default='mcs', help='dataset to train with')
parser.add_argument('--mcs_task', default='ObjectPermanenceTraining4', help='mcs task')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=45, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=50, help='number of frames to predict at eval time')
parser.add_argument('--start_min', type=int, default=65, help='min starting time for sampling sequence (0-indexed)')
parser.add_argument('--start_max', type=int, default=85, help='max starting time for sampling sequence  (0-indexed)')
parser.add_argument('--sequence_stride', type=int, default=3, help='factor for sequence temporal subsampling (int)')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=32, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--gamma', type=float, default=0.0001, help='weighting on h vs h posterior')
parser.add_argument('--model', default='vgg', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=12, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+1 rather than last ground truth frame')


opt = parser.parse_args()
saved_model = None
if opt.model_dir != '':
    models = glob.glob(f'{opt.model_dir}/model_*.pth')
    latest_model = sorted(models, key=lambda s: int(s[s.rfind('_e') + 2: s.rfind('.pth')]), reverse=True)[0]
    print('Loading model ', latest_model)
    saved_model = torch.load(latest_model)
    model_dir = opt.model_dir
    niter = opt.niter
    lr = opt.lr
    batch_size = opt.batch_size
    n_future = opt.n_future
    n_eval = opt.n_eval
    data_root = opt.data_root
    opt = saved_model['opt']
    opt.niter = niter  # update number of epochs to train for
    opt.model_dir = model_dir
    opt.n_future = n_future
    opt.lr = lr
    opt.batch_size = batch_size
    opt.n_eval = n_eval
    opt.data_root = data_root
    opt.log_dir = '%s/continued_lr%s' % (opt.log_dir, opt.lr)
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%d-beta=%.7f-gamma=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.gamma, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    elif opt.dataset == 'mcs':
        opt.log_dir = '%s/%s/%s/%s' % (opt.log_dir, opt.dataset, opt.mcs_task, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
with open(os.path.join(opt.log_dir, 'opt.json'), 'w') as f:
    opt2 = opt.__dict__.copy()
    if isinstance(opt2['optimizer'], type):
        opt2['optimizer'] = str(opt2['optimizer'])
    json.dump(opt2, f, indent=2)
    del opt2

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
elif isinstance(opt.optimizer, type):
    pass
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    frame_predictor.batch_size = opt.batch_size
    prior = saved_model['prior']
    prior.batch_size = opt.batch_size
    posterior = saved_model['posterior']
    posterior.batch_size = opt.batch_size
else:
    frame_predictor = lstm_models.lstm(opt.g_dim + opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior = lstm_models.lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    prior = lstm_models.lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

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
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, 1)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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
prior.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True,)
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
    gt_seq = [utils.torch_rgb_img_to_gray(x[t]) for t in range(len(x))]

    # h_seq = [encoder(x[i]) for i in range(opt.n_past)]
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(utils.torch_rgb_img_to_gray(x[0]))
        x_in = x[0]
        for i in range(1, opt.n_eval):
            with torch.no_grad():
                # if input is grayscale
                if x_in.shape[1] == 1 and opt.channels == 3:
                    h = encoder(torch.cat(3*[x_in], dim=1))  # convert to RGB
                else:
                    h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])
                z_t = posterior(h_target[0].detach())
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(utils.torch_rgb_img_to_gray(x_in))
            else:
                z_t_hat = prior(h)
                h = frame_predictor(torch.cat([h, z_t_hat], 1)).detach()
                x_in = decoder([h, skip])
                gen_seq[s].append(x_in)


    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
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
    gen_seq = [utils.torch_rgb_img_to_gray(x[0])]
    gen_seq_post = [utils.torch_rgb_img_to_gray(x[0])]

    # prediction using posterior Z
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    h = encoder(x[0])
    h = (h[0].detach(), h[1])
    for i in range(1, min(opt.n_eval, opt.n_past+opt.n_future)):
        h_target = encoder(x[i])
        h_target = (h_target[0].detach(), h_target[1])
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        z_t = posterior(h_target[0])
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1))
            gen_seq_post.append(utils.torch_rgb_img_to_gray(x[i]))
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq_post.append(x_pred)
        h = h_target

    # prediction using prior Z
    frame_predictor.hidden = frame_predictor.init_hidden()
    prior.hidden = prior.init_hidden()
    h = encoder(x[0])
    h = (h[0].detach(), h[1])
    for i in range(1, min(opt.n_eval, opt.n_past+opt.n_future)):
        h_target = encoder(x[i])
        h_target = (h_target[0].detach(), h_target[1])
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        z_t_hat = prior(h)
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t_hat], 1))
            gen_seq.append(utils.torch_rgb_img_to_gray(x[i]))
        else:
            h_pred = frame_predictor(torch.cat([h, z_t_hat], 1)).detach()
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)
        h = h_target
   
    to_plot = []
    nrow = min(opt.batch_size, 25)
    x_gray = [utils.torch_rgb_img_to_gray(x[t]) for t in range(min(opt.n_eval, opt.n_past+opt.n_future))]
    for i in range(nrow):
        row_gt = []
        row_post = []
        row = []
        for t in range(min(opt.n_eval, opt.n_past+opt.n_future)):
            row_gt.append(x_gray[t][i])
            row_post.append(gen_seq_post[t][i])
            row.append(gen_seq[t][i])
        to_plot.append(row_gt)
        to_plot.append(row_post)
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# --------- training funtions ------------------------------------
def train(x):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    mse = 0
    mse_residual = 0
    # x: T x B x C x H x W
    x_diff = [1]
    for i in range(1, len(x)):
        diff = torch.abs(x[i] - x[i - 1])
        diff = torch.mean(diff, dim=(1, 2, 3)).detach()  # mean over channels, width, and height
        x_diff.append(diff)

        # print(i, x_diff > 1e-6)
    h = encoder(x[0])
    for i in range(1, opt.n_past+opt.n_future):
        h_target = encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h = h[0]

        z_t = posterior(h_target[0])
        z_t_hat = prior(h)
        h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred = decoder([h_pred, skip])
        gray_target_frame = utils.torch_rgb_img_to_gray(x[i])

        still_frames = x_diff[i] <= 5e-6
        weights = torch.pow(0.05, still_frames).detach()[:, None, None, None]
        mse += mse_criterion(weights * x_pred, weights * gray_target_frame)
        # penalize prior for being far from posterior
        mse_residual += opt.gamma * torch.mean(weights * torch.square(z_t.detach() - z_t_hat))
        h = h_target

    loss = mse + mse_residual
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    N = opt.n_past+opt.n_future
    return mse.data.cpu().numpy()/N, mse_residual.data.cpu().numpy()/N

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_mse_residual = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    # opt.epoch_size = 10
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        # train frame_predictor 
        mse, mse_residual = train(x)
        epoch_mse += mse
        epoch_mse_residual += mse_residual


    progress.finish()
    utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | residual mse: %.20f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_mse_residual/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # plot some stuff
    frame_predictor.eval()
    posterior.eval()
    prior.eval()
    encoder.eval()
    decoder.eval()
    x = next(testing_batch_generator)
    plot(x, epoch)
    plot_rec(x, epoch)

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'opt': opt},
        '%s/model_e%02d.pth' % (opt.log_dir, epoch))
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        

