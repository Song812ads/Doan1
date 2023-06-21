import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATLayerEdgeAverage(nn.Module):
    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerEdgeAverage,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = eps
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        a = self.w(h) # E,1
        a_sum = torch.mm(Mtgt,a) + self.eps # N,E x E,1 = N,1
        o = torch.mm(Mtgt,y * a) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o

class GATLayerEdgeSoftmax(nn.Module):
    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerEdgeSoftmax,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = eps
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        a = self.w(h) # E,1
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a,0,keepdim=True)#[0] + self.eps
        assert not torch.isnan(a_base).any()
        a_norm = a-a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.mm(Mtgt,a_exp) + self.eps # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.mm(Mtgt,y * a_exp) / a_sum # N,1
        assert not torch.isnan(o).any()
        return o

class GATLayerMultiHead(nn.Module):
    
    def __init__(self,d_in,d_out,num_heads):
        super(GATLayerMultiHead,self).__init__()
        
        self.GAT_heads = nn.ModuleList(
              [
                GATLayerEdgeAverage(d_in,d_out)
                for _ in range(num_heads)
              ]
        )
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        return torch.cat([l(x,adj,src,tgt,Msrc,Mtgt) for l in self.GAT_heads],dim=1)


class GAT_final(nn.Module):
    
    def __init__(self,num_features,num_classes,num_heads=[3,3,3],layer1 = [32,64,64],dropout1 = 0.5):
        super(GAT_final,self).__init__()
        self.layer_heads = [1]+num_heads
        self.GAT_layer_sizes = [num_features]+layer1
        self.MLP_layer_sizes = [self.layer_heads[-1]*self.GAT_layer_sizes[-1],32,num_classes]
        self.MLP_acts = [F.relu,F.relu, lambda x:x]
        
        self.GAT_layers = nn.ModuleList(
              [
                GATLayerMultiHead(d_in*heads_in,d_out,heads_out)
                for d_in,d_out,heads_in,heads_out in zip(
                    self.GAT_layer_sizes[:-1],
                    self.GAT_layer_sizes[1:],
                    self.layer_heads[:-1],
                    self.layer_heads[1:],
                )
              ]
        )
        self.MLP_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.Dropout(p=dropout1),
                )
                for d_in, d_out in zip(self.MLP_layer_sizes[:-1], self.MLP_layer_sizes[1:])
            ]
        )
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt,Mgraph):
        for l in self.GAT_layers:
            x = l(x,adj,src,tgt,Msrc,Mtgt)
        x = torch.mm(Mgraph.t(),x)
        for layer,act in zip(self.MLP_layers,self.MLP_acts):
            x = act(layer(x))
        return torch.sigmoid(x)
        