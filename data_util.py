import gzip
import os
import urllib.request as request
from os import path
import random
from PIL import Image

import numpy as np

DATASET_DIR = "datasets/"

MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

NAMES_FILE = "names.txt"


def download_file(url, local_path):
    dir_path = path.dirname(local_path)
    if not path.exists(dir_path):
        print("Creating the directory '%s' ..." % dir_path)
        os.makedirs(dir_path)

    print("Downloading from '%s' ..." % url)
    request.urlretrieve(url, local_path)


def download_mnist(local_path):
    url_root = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    for f_name in MNIST_FILES:
        f_path = os.path.join(local_path, f_name)
        if not path.exists(f_path):
            download_file(url_root + f_name, f_path)


def one_hot(x, n):
    if isinstance(x, list):
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def load_mnist(ntrain=60000, ntest=10000, onehot=True):
    data_dir = os.path.join(DATASET_DIR, "mnist/")
    if not path.exists(data_dir):
        download_mnist(data_dir)
    else:
        # check all files
        checks = [path.exists(os.path.join(data_dir, f)) for f in MNIST_FILES]
        if not np.all(checks):
            download_mnist(data_dir)

    with gzip.open(os.path.join(data_dir, "train-images-idx3-ubyte.gz")) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28 * 28)).astype(np.float32)

    with gzip.open(os.path.join(data_dir, "train-labels-idx1-ubyte.gz")) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trY = loaded[8:].reshape((60000))

    with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 * 28)).astype(np.float32)

    with gzip.open(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teY = loaded[8:].reshape((10000))

    trX /= 255.0
    teX /= 255.0

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


def save_image_grid(x, file_path, nrow):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    x = np.array(x)
    # x = np.clip(x + 0.5, 0.0, 1.0)  # shift from [-0.5, 0.5] to [0.0, 1.0]
    x = np.clip(x * 0.5 + 0.5, 0.0, 1.0)  # shift from [-1.0, 1.0] to [0.0, 1.0]
    B, H, W, C = x.shape
    ncol = (B + nrow - 1) // nrow
    
    pad_B = nrow * ncol - B
    if pad_B > 0:
        x = np.pad(x, ((0, pad_B), (0, 0), (0, 0), (0, 0)))

    x = x.reshape((nrow, ncol, H, W, C))
    x = np.transpose(x, (0, 2, 1, 3, 4))
    grid = x.reshape((nrow * H, ncol * W, C))
    grid = (grid * 255.0).astype(np.uint8)  # convert back to [0, 255]

    if C == 1:
        grid = grid.squeeze(-1)
        img = Image.fromarray(grid, mode="L")
    elif C == 3:
        img = Image.fromarray(grid, mode="RGB")
    else:
        raise ValueError(f"Unsupported number of channels: {C}")
        
    img.save(file_path)


def download_names(local_path):
    url_root = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/'
    f_path = os.path.join(local_path, NAMES_FILE)
    if not path.exists(f_path):
        download_file(url_root + NAMES_FILE, f_path)


def load_names(rand_seed=42):
    data_dir = os.path.join(DATASET_DIR, "names/")
    if not path.exists(data_dir):
        download_names(data_dir)

    with open(os.path.join(data_dir, NAMES_FILE)) as f:
        lines = f.read().strip().split('\n')
        docs = [l.strip() for l in lines if l.strip()]

    random.seed(rand_seed)
    random.shuffle(docs)
    return docs
