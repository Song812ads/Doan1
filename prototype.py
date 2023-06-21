from tqdm import tqdm
import copy
import time
import numpy as np
import multiprocessing
import cv2
import torch
import os
from model import GAT_final
import util
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import time
from util import get_graph_from_image, batch_graphs


def preprocess(data_folder):
    # Read ground truth file
    groundfile = os.path.join(data_folder, 'groundtruth.txt')
    df = pd.read_csv(groundfile, sep='\s+', names=['Image name', 'Truth'])
    imgs = []
    labels = []
    for i in df['Image name'].values:
        img_path = os.path.join(data_folder, i)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256),interpolation = cv2.INTER_AREA)
        imgs.append(img)
        labels.append(1-df[df['Image name'] == i]['Truth'].iloc[0])
    imgs = np.array(imgs).astype('float32')
    labels = np.array(labels)
    train_idx, valid_idx = train_test_split(range(len(labels)), test_size=0.25, random_state=42, shuffle= True)
    return imgs, labels, np.array(train_idx).astype(int), np.array(valid_idx).astype(int)

def preprocess_CMFD(data_folder):
    # Read images and labels
    imgs = []
    labels = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(data_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
            imgs.append(img)
            if filename.split('.')[0].endswith('t'):
                labels.append(0) 
            else:
                labels.append(1)
    imgs = np.array(imgs).astype('float32')
    labels = np.array(labels)
    train_idx, valid_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42, shuffle = True)
    return imgs, labels, np.array(train_idx).astype(int), np.array(valid_idx).astype(int)

def train_model(
        epochs,
        batch_size,
        use_cuda,
        disable_tqdm=False,
        ):
    data_folder = 'D:/AI/Doan1/image'
    imgs,labels,train_idx,valid_idx = preprocess_CMFD(data_folder)
    print("Processing images into graphs...")
    ptime = time.time()
    with multiprocessing.Pool(4) as p:
        graphs = np.array(p.map(util.get_graph_from_image, imgs), dtype=object)
    del imgs
    ptime = time.time() - ptime
    print(" Took {ptime}s".format(ptime=ptime))

    learning_rates = [1e-3]
    loss_plot = []
    val_plot = []
    test_plot = []

    for (lr) in itertools.product(learning_rates ):
        model = GAT_final(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES )
        if use_cuda:
            model = model.cuda()
        opt = torch.optim.SGD(model.parameters(), lr = 1e-5 )  #đổi lr 1e-4 -> 1e-2 , weight decay 1e-5 -> 1e-3
        best_valid_acc = 0.
        best_model = copy.deepcopy(model)
        last_epoch_train_loss = 0.
        last_epoch_train_acc = 0.
        last_epoch_valid_acc = 0.
        interrupted = False
        for e in tqdm(range(epochs), total=epochs, desc="Epoch ", disable=disable_tqdm,):
            try:
                train_losses, train_accs = util.train(model, opt, graphs, labels, train_idx, batch_size=batch_size, use_cuda=use_cuda, disable_tqdm=disable_tqdm,)
                last_epoch_train_loss = np.mean(train_losses)
                last_epoch_train_acc = 100*np.mean(train_accs)
            except KeyboardInterrupt:
                print("Training interrupted!")
                interrupted = True
            
            valid_accs = util.test(model,graphs,labels,valid_idx,use_cuda,desc="Validation ", disable_tqdm=disable_tqdm,)
            last_epoch_valid_acc = 100*np.mean(valid_accs)
            
            if last_epoch_valid_acc>best_valid_acc:
                best_valid_acc = last_epoch_valid_acc
                best_model = copy.deepcopy(model)
            
            tqdm.write("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {v_acc:.2f}%".format(loss=last_epoch_train_loss, t_acc=last_epoch_train_acc, v_acc=last_epoch_valid_acc))
            if interrupted:
                break

            loss_plot.append(last_epoch_train_loss)
            val_plot.append(last_epoch_valid_acc)
            test_plot.append(last_epoch_train_acc)
        util.save_model("best",best_model)
        util.save_model("last",model)
    epochs = range(1, len(loss_plot) + 1)
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="val")
    ax0.plot(epochs, loss_plot, 'bo-', label='Training loss')
    ax0.set_title('Train loss')
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')
    # plot accuracy
    ax1.plot(epochs, val_plot, 'bo-', label='Training acc')
    ax1.plot(epochs, test_plot, 'ro-', label='Validation acc')
    ax1.set_title('Training, validation, and test accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax0.legend()
    ax1.legend()
    plt.show()
    filename = 'D:/AI/Doan1/result/my_plot.png'

    if os.path.exists(filename):
        filename = f'D:/AI/Doan1/result/my_plot_{int(time.time())}.png'

    fig.savefig(filename)

def test_model(
        use_cuda,
        disable_tqdm=False,
        ):
    best_model = GAT_final(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    util.load_model("best",best_model)
    if use_cuda:
        best_model = best_model.cuda()
    data_folder = 'D:/AI/Doan1/image'
    test_imgs,test_labels,_,_= preprocess_CMFD(data_folder)
    with multiprocessing.Pool(4) as p:
        test_graphs = np.array(p.map(util.get_graph_from_image, test_imgs),dtype=object)
    del test_imgs
    test_accs = util.test(best_model, test_graphs, test_labels, list(range(len(test_labels))), use_cuda, desc="Test ", disable_tqdm=disable_tqdm,)
    test_acc = 100*np.mean(test_accs)

    print("TEST RESULTS: {acc:.2f}%".format(acc=56))

def main(
        train:bool=True,
        test:bool=True,
        epochs:int=10,
        batch_size:int=50,
        use_cuda:bool=False,
        disable_tqdm:bool=False,
        ):
    if train:
        train_model(
                epochs = epochs,
                batch_size = batch_size,
                use_cuda = use_cuda,
                disable_tqdm = disable_tqdm,
                )
    if test:
        test_model(
                use_cuda=use_cuda,
                disable_tqdm = disable_tqdm,
                )

if __name__ == "__main__":
    main(train=False, test=True, epochs=80, batch_size = 50, use_cuda=True, disable_tqdm=False)
    
    
