import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import numpy as np
import skimage.io
import skimage.transform
import os

# http://yann.lecun.com/exdb/mnist/


def convert(imgf, labelf, ofp, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    if not os.path.exists(ofp):
        os.mkdir(ofp)

    f.read(16)
    l.read(8)

    for i in range(n):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        img = np.asarray(image, dtype=np.uint8)
        img = img.reshape((28,28))
        img = img.astype(np.uint8)

        skimage.io.imsave(os.path.join(ofp, 'img_{:08d}_{}.tif'.format(i, int(ord(l.read(1))))), img, compress=6)
    f.close()
    l.close()


fp = '/scratch/datasets/MNIST/'

convert(os.path.join(fp, 'train-images-idx3-ubyte'), os.path.join(fp, 'train-labels-idx1-ubyte'), os.path.join(fp, 'train'), 60000)
convert(os.path.join(fp, 't10k-images-idx3-ubyte'), os.path.join(fp, 't10k-labels-idx1-ubyte'), os.path.join(fp, 'test'), 10000)
