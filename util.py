from tqdm import tqdm
import time
import numpy as np
import scipy as sp
from skimage.segmentation import slic
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import cv2
import skimage.feature as skif
from skimage import io, color
from numpy.linalg import norm
import skimage
from skimage.feature import local_binary_pattern    
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int32

NUM_FEATURES = 9
NUM_CLASSES = 1

    
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

def get_graph_from_image(image,desired_nodes=80):
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

def batch_graphs(gs):

    G = len(gs)
    N = sum(g[0].shape[0] for g in gs)
    M = sum(g[1].shape[0] for g in gs)
    adj = np.zeros([N,N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N,M])
    Mtgt = np.zeros([N,M])
    Mgraph = np.zeros([N,G])
    h = np.concatenate([g[0] for g in gs])
    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = g[0].shape[0]
        m = g[1].shape[0]
        for e,(s,t) in enumerate(g[1]):
            adj[n_acc+s,n_acc+t] = 1
            adj[n_acc+t,n_acc+s] = 1
            src[m_acc+e] = n_acc+s
            tgt[m_acc+e] = n_acc+t
            Msrc[n_acc+s,m_acc+e] = 1
            Mtgt[n_acc+t,m_acc+e] = 1
        Mgraph[n_acc:n_acc+n,g_idx] = 1
        n_acc += n
        m_acc += m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )
    
def save_model(fname, model):
    torch.save(model.state_dict(),"{fname}.pt".format(fname=fname))
    
def load_model(fname, model):
    model.load_state_dict(torch.load("{fname}.pt".format(fname=fname)))

def to_cuda(x):
    return x.cuda()
def split_dataset(labels, valid_split=0.1):
    idx = np.random.permutation(len(labels))
    valid_idx = []
    train_idx = []
    label_count = [0 for _ in range(1+max(labels))]
    valid_count = [0 for _ in label_count]
    
    for i in idx:
        label_count[labels[i]] += 1
    
    for i in idx:
        l = labels[i]
        if valid_count[l] < label_count[l]*valid_split:
            valid_count[l] += 1
            valid_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, valid_idx

def train(model, optimiser, graphs, labels, train_idx, use_cuda = False, batch_size=1, disable_tqdm=False, profile=False):
    train_losses = []
    train_accs = []
    indexes = train_idx[np.random.permutation(len(train_idx))]
    pyt_labels = torch.tensor(labels)
    if use_cuda:
        pyt_labels = pyt_labels.cuda()
    for b in tqdm(range(0,len(indexes),batch_size), total=len(indexes)/batch_size, desc="Instances ", disable=disable_tqdm):
        ta = time.time()
        optimiser.zero_grad()
        batch_indexes = indexes[b:b+batch_size]
        batch_labels = pyt_labels[batch_indexes]
        tb = time.time()
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs[batch_indexes])
        tc = time.time()
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.from_numpy,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
        td = time.time()
        if use_cuda:
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
        te = time.time()
        y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
        tf = time.time()
        lost = nn.BCELoss()
        batch_labels = batch_labels.detach().cpu().numpy()
        batch_labels = torch.from_numpy(batch_labels).float().reshape(-1, 1)
        if use_cuda:
            y = y.cpu()
            batch_labels = batch_labels.to(y.device)
        loss = lost(y,batch_labels)
        pred = torch.round(y).detach().cpu().numpy()
        acc = np.sum((pred == batch_labels.numpy()).astype(float)) / batch_labels.shape[0]
        mode = sp.stats.mode(pred,keepdims = True)
        tg = time.time()
        tqdm.write(
              "{loss:.4f}\t{acc:.2f}%\t{mode} (x{modecount})".format(
                  loss=loss.item(),
                  acc=100*acc,
                  mode=mode[0][0],
                  modecount=mode[1][0],
              )
        )
        th = time.time()
        loss.backward()
        optimiser.step()
        train_losses.append(loss.detach().cpu().item())
        train_accs.append(acc)
        if profile:
            ti = time.time()
            tt = ti-ta
            tqdm.write("zg {zg:.2f}% bg {bg:.2f}% tt {tt:.2f}% tc {tc:.2f}% mo {mo:.2f}% me {me:.2f}% bk {bk:.2f}%".format(
                    zg=100*(tb-ta)/tt,
                    bg=100*(tc-tb)/tt,
                    tt=100*(td-tc)/tt,
                    tc=100*(te-td)/tt,
                    mo=100*(tf-te)/tt,
                    me=100*(tg-tf)/tt,
                    bk=100*(ti-th)/tt,
                    ))
    return train_losses, train_accs

def test(model, graphs, labels, indexes, use_cuda, desc="Test ", disable_tqdm=False):
    test_accs = []
    for i in tqdm(range(len(indexes)), total=len(indexes), desc=desc, disable=disable_tqdm):
        with torch.no_grad():
            idx = indexes[i]
            batch_labels = labels[idx:idx+1]
            pyt_labels = torch.from_numpy(batch_labels)
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs([graphs[idx]])
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.from_numpy,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
            if use_cuda:
                h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels))
            y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
            pred = torch.round(y).detach().cpu().numpy()
            acc = np.sum((pred==batch_labels).astype(float)) / batch_labels.shape[0]
            test_accs.append(acc)
    return test_accs


