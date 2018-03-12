# Stochastic Video Generation with a Learned Prior
This is code for the paper [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687) by Emily Denton and Rob Fergus. See the [project page](https://sites.google.com/view/svglp/) for details and generated video sequences.

##  Training on Stochastic Moving MNIST (SM-MNIST)
To train the SVG-LP model on the 2 digit SM-MNIST dataset run: 
```
python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```
If the MNIST dataset doesn't exist, it will be downloaded to the specified path.

## BAIR robot push dataset
To download the BAIR robot push dataset run:
```
sh data/download_bair.sh /path/to/data/
```
This will download the dataset in tfrecord format into the specified directory. To train the pytorch models, we need to first convert the tfrecord data into .png images by running:
```
python data/convert_bair.py --data_dir /path/to/data/
```
This may take some time. Images will be saved in ```/path/to/data/processeddata```.
Now we can train the SVG-LP model by running:
```
python train_svg_lp.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```

To generate images with a pretrained SVG-LP model run:
```
python generate_svg_lp.py --model_path pretrain_models/svglp_bair.pth --log_dir /generated/images/will/save/here/
```
