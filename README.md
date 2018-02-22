# Stochastic Video Generation with a Learned Prior
This is code for the paper [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687) by Emily Denton and Rob Fergus. See the [project page](https://sites.google.com/view/svglp/) for details and generated video sequences.

##  Training on Stochastic Moving MNIST (SM-MNIST)
To train the SVG-LP model on the 2 digit SM-MNIST dataset run: 
```
python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```
If the MNIST dataset doesn't exist, it will be downloaded to the specified path.
