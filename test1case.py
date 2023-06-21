from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import torch
import numpy as np
import math
from math import sqrt
from scipy import spatial
from skimage.metrics import structural_similarity as ssim
import numpy as np
from numpy.linalg import norm
from scipy.special import softmax
import statistics 
import pandas as pd
import networkx as nx
import sys
pic_1 = "D:/AI/doan1/test.jpg"
from numpy import random
import torch.nn.functional as F
from scipy.special import softmax
from sklearn import preprocessing
from skimage.color import rgb2gray
from skimage.segmentation import relabel_sequential
from skimage import feature as skif

image = cv2.imread(pic_1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (80, 80),interpolation = cv2.INTER_AREA)
slic_arr = slic(image, n_segments=100, sigma=2.1, slic_zero = True)
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, slic_arr))
plt.show()