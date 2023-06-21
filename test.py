from tqdm import tqdm
import util
import skimage
import time
import pickle
import multiprocessing
import cv2
import numpy as np
import scipy as sp
from skimage.segmentation import slic, mark_boundaries
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.linalg import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from sklearn import preprocessing
from model import GAT_final
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer,    RobustScaler, MaxAbsScaler
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64
from skimage.feature import local_binary_pattern
NUM_FEATURES = 9
NUM_CLASSES = 1
import sys
    
def rgb_histogram(image,part):
    r_hist1, _ = np.histogram(image[:, :, 0].ravel()[part], bins=8, range=(0, 255)) 
    r_hist1 = r_hist1 + 1e-6
    r_hist1 = r_hist1 / np.linalg.norm(r_hist1)
    
    g_hist1, _ = np.histogram(image[:, :, 1].ravel()[part], bins=8, range=(0, 255)) 
    g_hist1 = g_hist1 + 1e-6
    g_hist1 = g_hist1 / np.linalg.norm(g_hist1)
    
    b_hist1, _ = np.histogram(image[:, :, 2].ravel()[part], bins=8, range=(0, 255)) 
    b_hist1 = b_hist1 + 1e-6
    b_hist1 = b_hist1 / np.linalg.norm(b_hist1)
    hist1 = np.concatenate((r_hist1, g_hist1, b_hist1))
    return hist1 

def get_graph_from_image(image,desired_nodes=100):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_bgr = cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
    segments = slic(image, n_segments=desired_nodes, sigma=2.1, slic_zero = True)
    asegments = np.array(segments)
    num_nodes = np.max(asegments)
    nodes = {
        node: {
            "lab_list": [],
            "pos_list": [],
            "gray_list": [],
        } for node in range(num_nodes)
    }

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = asegments[y,x]
            rgb = image[y,x,:]
            gray = image_gray[y,x]
            pos = np.array([float(x)/width,float(y)/height])
            nodes[node-1]["lab_list"].append(rgb)
            nodes[node-1]["pos_list"].append(pos)
            nodes[node-1]["gray_list"].append(gray)
    ls_mean = []
    as_mean = []
    bs_mean = []
    constrast = []
    correlation = []
    homogenity = []
    areas = []
    perimeters = []
    compactnesss = []
    aspect_ratios = []
    energy = []

    G = nx.Graph()
    for node in nodes:
        nodes[node]["lab_list"] = np.stack(nodes[node]["lab_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        nodes[node]["gray_list"] = np.stack(nodes[node]["gray_list"])
        l_mean, a_mean, b_mean = np.mean(nodes[node]["lab_list"], axis=0)
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        ls_mean.append(np.reshape(l_mean,-1))
        as_mean.append(np.reshape(a_mean,-1))
        bs_mean.append(np.reshape(b_mean,-1))
        gray_list = nodes[node]["gray_list"]
        gray_list = np.round(gray_list * 255).astype(np.uint8)
        gray_list = np.reshape(gray_list, (1, len(gray_list)))
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        mask[segments == node+1] = 255
        moments = cv2.moments(mask)
        area = moments['m00']/255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (perimeter**2) / area
        aspect_ratio = moments['mu20'] / moments['mu02']
        areas.append(np.reshape(area,-1))
        perimeters.append(np.reshape(perimeter,-1))
        compactnesss.append(np.reshape(compactness,-1))
        aspect_ratios.append(np.reshape(aspect_ratio,-1))


        co_matrix = skimage.feature.graycomatrix(gray_list, [5], [0], levels=256, symmetric=True, normed=True)
        constrast.append(np.reshape(skimage.feature.graycoprops(co_matrix, 'contrast'),-1))
        correlation.append(np.reshape(skimage.feature.graycoprops(co_matrix, 'correlation'),-1))
        energy.append(np.reshape(skimage.feature.graycoprops(co_matrix, 'energy'),-1))
        homogenity.append(np.reshape(skimage.feature.graycoprops(co_matrix, 'homogeneity'),-1))
        G.add_node(node, pos = np.reshape(pos_mean,-1))
    # print(texture_features)
    scaler = MinMaxScaler(feature_range= (-1,1), copy = False, clip = True)
    ls_mean = scaler.fit_transform(np.array(ls_mean))
    as_mean = scaler.fit_transform(np.array(as_mean))
    bs_mean = scaler.fit_transform(np.array(bs_mean))
    constrast = scaler.fit_transform(np.array(constrast))
    homogenity = scaler.fit_transform(np.array(homogenity))
    correlation = scaler.fit_transform(np.array(correlation))
    energy = scaler.fit_transform(np.array(energy))
    areas = scaler.fit_transform(np.array(areas))
    perimeters = scaler.fit_transform(np.array(perimeters))
    compactnesss = scaler.fit_transform(np.array(compactnesss))
    aspect_ratios = scaler.fit_transform(np.array(aspect_ratios))


    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0,i] != bneighbors[1,i]:
            node1 = bneighbors[0,i] - 1
            node2 = bneighbors[1,i] - 1 
            src = np.where(segments.ravel() == node1)[0]
            hist1 = rgb_histogram(image,src)
            tgt = np.where(segments.ravel() == node2)[0]
            hist2 = rgb_histogram(image,tgt) 
            
            if np.dot(hist1, hist2) / (norm(hist1) * norm(hist2)) > 0.5 :
                G.add_edge(bneighbors[0,i]-1,bneighbors[1,i]-1)
        

    # Self loops
    for node in nodes:
        G.add_edge(node,node)
    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n,NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2*m,2]).astype(NP_TORCH_LONG_DTYPE)
    for e,(s,t) in enumerate(G.edges):
        edges[e,0] = s
        edges[e,1] = t
        edges[m+e,0] = t
        edges[m+e,1] = s
    for i in G.nodes:
        feat = np.concatenate((
                            ls_mean[i], as_mean[i], bs_mean[i], 
                           constrast[i],homogenity[i], correlation[i], energy[i],
                            # areas[i], perimeters[i], compactnesss[i], aspect_ratios[i],
                            np.array(G.nodes[node]['pos'])
        ))
        h[i,:] = feat
        # h[i,:] = np.array(G.nodes[node]['pos'])
    del G
    return h, edges

def batch_graphs(g):
    NUM_FEATURES = g[0].shape[-1]
    G = 1

    N = g[0].shape[0]
    M = g[1].shape[0]
    adj = np.zeros([N,N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N,M])
    Mtgt = np.zeros([N,M])
    Mgraph = np.zeros([N,G])
    h = g[0]

    for e, (s,t) in enumerate(g[1]):
        adj[s,t] = 1
        adj[t,s] = 1

        src[e] = s
        tgt[e] = t

        Msrc[s,e] = 1
        Mtgt[t,e] = 1

    Mgraph[:,0] = 1
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )


if __name__ == "__main__":
    pic_1 = "D:/AI/doan1/fake.jpg"
    image = cv2.imread(pic_1)
    image = cv2.resize(image,(256,256))
    graphs = get_graph_from_image(image)
    h,adj,src,tgt,MSrc,Mtgt,Mgraph = batch_graphs(graphs)
    adj = torch.from_numpy(adj)
    src = torch.from_numpy(src)
    tgt = torch.from_numpy(tgt)
    Msrc = torch.from_numpy(MSrc)
    Mtgt = torch.from_numpy(Mtgt)
    Mgraph = torch.from_numpy(Mgraph)
    h = torch.from_numpy(h)
    model = GAT_final(9,1)
    util.load_model('D:/AI/Doan1/last',model)
    x = model(h,adj,src,tgt,MSrc,Mtgt,Mgraph = Mgraph)

    while True:
        cv2.imshow("Day la anh gia", image)
        cv2.waitKey(0)
        sys.exit() # to exit from all the processes
    
    cv2.destroyAllWindows()


    # while True:
    #     image = cv2.imread(pic_1)
    #     cv2.imshow("Day la anh gia", image)
    #     cv2.waitKey(0)
    #     sys.exit()