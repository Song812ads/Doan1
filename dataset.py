import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import scipy.sparse as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sp
import shutil
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


def make_dir(root_dir, target_dir, remove=False):
    result_dir = os.path.join(root_dir, target_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    elif remove == True:
        shutil.rmtree(result_dir)
        os.mkdir(result_dir)
    return result_dir

class Xuly_data(Dataset):
    def __init__(self, root_dir, n_superpix, mode):
        self.n_superpix = n_superpix
        self.image_paths = []
        img_dirs = os.path.join(root_dir,"casia")
        self.mode = mode
        self.root_dir = root_dir
# Tao 1 image_paths luu tat ca cac path dan den anh vao 
        if mode == 'train': 
            img_dir = os.path.join(img_dirs,"Au")
            soluonganh = len(glob.glob(os.path.join(img_dir,"*.jpg")))
            for _ in os.listdir(img_dir):
                path = os.path.join(img_dir,"*.jpg")
                if os.path.isfile(path):
                    self.image_paths.append(path)
        else: 
            img_dir = os.path.join(img_dirs,"Sp")
            soluonganh = len(glob.glob(os.path.join(img_dir,"*.jpg")))
            for _ in os.listdir(img_dir):
                path = os.path.join(img_dir,"*.jpg")
                if os.path.isfile(path):
                    self.image_paths.append(path)
            
# Getitem la 1 giao thuc chia data thanh cac lop nho hon de lay thong tin
    def __getitem__(self, idx): 
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        slic_arr = slic(image, n_segments=self.n_superpix,compactness = 10,sigma=1.0)
        image = image/255.0
        image = torch.FloatTensor(image)
        return image,slic_arr,img_name
    
    def __len__(self):
        return len(self.image_paths)
    
    def slic_image(self):
        len = self.__len__()
        image,slic_arr,img_name = self.__getitem__()
      ##  for _ in range(len): 