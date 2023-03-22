from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
from data_graph import show_graph_with_labels , gen_adj
import torch
import numpy as np
pic_1 = "D:/doan1/test.jpg"


image = cv2.imread(pic_1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(256,256))
slic_arr = slic(image, n_segments=256,  sigma=5.0)
_, indexs = np.unique(slic_arr[255], return_index=True)
size = slic_arr.shape
##print(size[1])
for i in range(size[0]):
# Cac diem quan trong hang i (node represent)
    _, indexs = np.unique(slic_arr[i], return_index=True)
    near_arr = [slic_arr[i][index] for index in sorted(indexs)]
 ##   print(len(near_arr))
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, slic_arr))
plt.axis("off")
plt.show()
image = image/255.0 
image = torch.FloatTensor(image)
row,col = np.where(slic_arr == 0)
adj = gen_adj(slic_arr)
##print(row,col)
show_graph_with_labels(adj,slic_arr)
