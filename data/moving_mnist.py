import socket
import numpy as np
from torchvision import datasets, transforms

class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True):
        path = '/misc/vlgscratch4/FergusGroup/denton/data/mnist/' 
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Scale(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) 

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x


class MovingMNISTSynced(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64):
        path = '/misc/vlgscratch4/FergusGroup/denton/data/mnist/' 
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Scale(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) 
        self.sx = np.random.randint(image_size-self.digit_size)
        self.sy = np.random.randint(image_size-self.digit_size)
        self.dx = [np.random.randint(-4, 5) for i in range(100)]
        self.dy = [np.random.randint(-4, 5) for i in range(100)]

        self.pos_dx = [np.random.randint(1, 5) for i in range(100)]
        self.pos_dy = [np.random.randint(1, 5) for i in range(100)]

        self.neg_dx = [np.random.randint(-4, 0) for i in range(100)]
        self.neg_dy = [np.random.randint(-4, 0) for i in range(100)]

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def get_sample(self, digit):
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      self.channels,
                      image_size, 
                      image_size, ),
                    dtype=np.float32)

        for n in range(self.num_digits):
            sx = self.sx
            sy = self.sy
            dx = self.dx[0]
            dy = self.dy[0]
            k = 1
            times = []
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    dy = np.random.randint(1, 5)
                    dx = np.random.randint(-4, 5)
                    k+=1
                    times.append(t)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    dy = np.random.randint(-4, 0)
                    dx = np.random.randint(-4, 0)
                    k+=1
                    times.append(t)
                    
                if sx < 0:
                    sx = 0 
                    dx = np.random.randint(1, 5)
                    dy = np.random.randint(-4, 5)
                    k+=1
                    times.append(t)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    dx = np.random.randint(-4, 0)
                    dy = np.random.randint(-4, 5)
                    k+=1
                    times.append(t)
                   
                x[t,0, sy:sy+32, sx:sx+32] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
        

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)

        all_times = []
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = self.sx
            sy = self.sy
            dx = self.dx[0]
            dy = self.dy[0]
            k = 1
            times = []
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    dy = self.pos_dy[k] #np.random.randint(1, 4)
                    dx = self.dx[k] #np.random.randint(-4, 4)
                    k+=1
                    times.append(t-1)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    dy = self.neg_dy[k+2*n] #np.random.randint(-4, -1)
                    dx = self.dx[k+2*n] #np.random.randint(-4, 4)
                    k+=1
                    times.append(t-1)
                    
                if sx < 0:
                    sx = 0 
                    dx = self.pos_dx[k+2*n] #np.random.randint(1, 4)
                    dy = self.dy[k+2*n] #np.random.randint(-4, 4)
                    k+=1
                    times.append(t-1)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    dx = self.neg_dx[k+2*n] #np.random.randint(-4, -1)
                    dy = self.dy[k+2*n] #np.random.randint(-4, 4)
                    k+=1
                    times.append(t-1)
                   
                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx
            all_times.append(times)

        x[x>1] = 1.

        x_sample = []
        for s in range(100):
            #x_sample.append(0)
            x_sample.append(self.get_sample(digit))

        return x, digit, np.array(all_times[0]), np.array(all_times[0]), np.array(x_sample)
