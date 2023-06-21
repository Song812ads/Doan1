import torch
import os
import numpy as np
import networkx as nx
import torch
from skimage.segmentation import slic
import cv2
import shutil
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import lightning as L
import torch.optim as optim
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import lightning as L
import os 
from sklearn.preprocessing import MinMaxScaler
import torch
from multiprocessing import Pool
CHECKPOINT_PATH = "D:/AI/Doan1/model"

def get_graph_from_image(image,desired_nodes=75):
    # load the image and convert it to a floating point data type
    segments = slic(image, n_segments=desired_nodes)
    asegments = np.array(segments)
    num_nodes = np.max(asegments)
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_nodes)
    }

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = asegments[y,x]
            rgb = image[y,x,:]
            pos = np.array([float(x)/width,float(y)/height])
            nodes[node-1]["rgb_list"].append(rgb)
            nodes[node-1]["pos_list"].append(pos)
        #end for
    #end for
    rgbs_mean = []
    rgbs_std = []
    rgbs_gram = []
    G = nx.Graph()
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        rgbs_mean.append(np.stack(np.reshape(rgb_mean, -1)))
        rgbs_std.append(np.stack(np.reshape(rgb_std, -1)))
        rgbs_gram.append(np.stack(np.reshape(rgb_gram, -1)))
        G.add_node(node, pos = np.reshape(pos_mean,-1))
    scaler = MinMaxScaler()
    rgbs_mean = scaler.fit_transform(np.array(rgbs_mean))

    h = np.zeros([desired_nodes,17]).astype('float32')

    for i in G.nodes:
        feat = np.concatenate((rgbs_mean[i],rgbs_std[i],rgbs_gram[i],np.array(G.nodes[node]['pos'])))
        h[i,:] = feat

    del G
    return h

class CustomDataset(Dataset):
    def __init__(self, dataset, df):
        self.dataset  = dataset
        self.labels = []
        for filename in self.dataset:
            label = df.loc[df['Image name'] == filename, 'Truth'].values[0]
            self.labels.append(label)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = os.path.join('D:/AI/doan1/dataset/',self.dataset[idx])
        image = cv2.imread(self.dataset[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(120,120))
        x = np.zeros((75,17))
        x = get_graph_from_image(image)
        x = torch.tensor(x).float()
        label = self.labels[idx]
        return x, label

# Load the dataframe
def preprocess():
    groundfile = os.path.join('D:/AI/doan1/dataset/groundtruth.txt')
    df = pd.read_csv(groundfile, sep='\s+', names=['Image name', 'Truth'])

    # Split the dataset into training and testing sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_imgs = []
    val_imgs = []
    # Copy images to train and val directories
    for i in train_df['Image name'].values[:100]:
        train_imgs.append(i)
    for i in val_df['Image name'].values[:100]:
        val_imgs.append(i)

    # Create the custom datasets and data loaders
    train_dataset = CustomDataset(train_imgs, train_df)
    val_dataset = CustomDataset(val_imgs, val_df)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers= 4)    
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers= 4)
    return train_loader,val_loader

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0): # fix dropout
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        node_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        # Layers/Networks
        self.input_layer = nn.Linear(self.node_dim, embed_dim) # change in_features
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
    def forward(self, x):
        # Preprocess input
        B, T, _ = x.shape
        x = x.view(B, T, self.node_dim) # concatenate node features
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        # Apply Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return torch.sigmoid(out)

class ViT(L.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds.float(),labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")
        
    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")        
  
def train_model(**kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"),
        accelerator="auto",
        devices=1,
        max_epochs=2,
        log_every_n_steps= 10,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = ViT.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}
    print("Validation accuracy:", result["val"])
    return model, result

if __name__ == "__main__":
    train_loader,val_loader = preprocess()
    model, results = train_model(
        model_kwargs={
            "embed_dim": 32,
            "hidden_dim": 128,
            "num_heads": 2,
            "num_layers": 2,
            "num_classes": 2,
            "node_dim":17,
            "dropout": 0.2,
        },
        lr=3e-4,
    )
    print("ViT results", results)  

