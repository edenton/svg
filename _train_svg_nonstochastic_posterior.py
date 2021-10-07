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

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.008, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/nonstochastic_posterior', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=400, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='mcs', help='dataset to train with')
parser.add_argument('--mcs_task', default='SpatioTemporalContinuityTraining4', help='mcs task')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=15, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--gamma', type=float, default=0.0001, help='weighting on h vs h posterior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
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
    opt = saved_model['opt']
    opt.niter = niter  # update number of epochs to train for
    opt.model_dir = model_dir
    opt.lr = lr
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
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior.apply(utils.init_weights)

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
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
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
            with torch.no_grad():
                h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h_seq[i-1]
            h = h.detach()
            if i < opt.n_past:
                frame_predictor(h)
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                h = frame_predictor(h).detach()
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
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()

    gen_seq = []
    gen_seq.append(x[0])
    gen_seq_post = []
    gen_seq_post.append(x[0])
    x_in = x[0]
    h_seq = [encoder(x[i]) for i in range(opt.n_past+opt.n_future)]
    for i in range(1, opt.n_past+opt.n_future):
        h_target = h_seq[i][0].detach()
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h_seq[i-1]
        else:
            h, _ = h_seq[i-1]
        h = h.detach()
        if i < opt.n_past:
            frame_predictor(h)
            gen_seq.append(x[i])
            gen_seq_post.append(x[i])
        else:
            h_pred = frame_predictor(h).detach()
            x_pred = decoder([h_pred, skip]).detach()
            h_posterior = posterior(h_target).detach()
            x_posterior = decoder([h_posterior, skip]).detach()
            gen_seq.append(x_pred)
            gen_seq_post.append(x_posterior)
   
    to_plot = []
    nrow = min(opt.batch_size * 3, 25 * 3)
    for i in range(nrow):
        row_gt = []
        row_post = []
        row = []
        for t in range(opt.n_past+opt.n_future):
            row_gt.append(x[t][i])
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
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()

    h_seq = [encoder(x[i]) for i in range(opt.n_past+opt.n_future)]
    mse = 0
    mse_post = 0
    mse_diff_post = 0
    for i in range(1, opt.n_past+opt.n_future):
        h_target = h_seq[i][0]
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h_seq[i-1]
        else:
            h = h_seq[i-1][0]
        h_pred = frame_predictor(h)
        x_pred = decoder([h_pred, skip])
        h_posterior = posterior(h_target)
        x_posterior = decoder([h_posterior, skip])
        mse += mse_criterion(x_pred, x[i])
        mse_post += mse_criterion(x_posterior, x[i])
        mse_diff_post += opt.gamma * torch.mean(torch.square(h_posterior.detach() - h_pred))

    loss = mse + mse_post + mse_diff_post
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    N = opt.n_past+opt.n_future
    return mse.data.cpu().numpy()/N, mse_post.data.cpu().numpy()/N, mse_diff_post.data.cpu().numpy()/N

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_mse_posterior = 0
    epoch_posterior_diff = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    # opt.epoch_size = 10
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        # train frame_predictor 
        mse, mse_posterior, posterior_diff = train(x)
        epoch_mse += mse
        epoch_mse_posterior += mse_posterior
        epoch_posterior_diff += posterior_diff


    progress.finish()
    utils.clear_progressbar()

    print('[%02d] mse loss: %.5f, %.5f posterior | posterior diff loss: %.20f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_mse_posterior/opt.epoch_size, epoch_posterior_diff/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # plot some stuff
    frame_predictor.eval()
    posterior.eval()
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
        'opt': opt},
        '%s/model_e%02d.pth' % (opt.log_dir, epoch))
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        

