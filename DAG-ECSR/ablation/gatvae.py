from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch .nn.functional as F
from ablation.gnn_utils import *

class CAVAE_gat(nn.Module):
    def __init__(self,args):
        super(CAVAE_gat,self).__init__()
        self.args=args
        self.encoder=GATVAEencoder(args)
        self.decoder=GATVAEdecoder(args)
    
    def forward(self, doc_sents_h, doc_len, adj):
        fU,adjB_list=self.encoder(doc_sents_h,doc_len,adj)
        X=self.decoder(fU,doc_len,adj,adjB_list)

        return X,adjB_list
        
class GATVAEencoder(nn.Module):
    def __init__(self, args):
        super(GATVAEencoder, self).__init__()
        in_dim = args.emb_dim
        self.gnn_dims = [in_dim,args.gat_feat_dim]

        self.gnn_layers = 1
        self.att_heads = [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GATencoderlayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], 0.1)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h,W = gnn_layer(doc_sents_h, adj)

        return doc_sents_h,W



class GATVAEdecoder(nn.Module):
    def __init__(self, args):
        super(GATVAEdecoder, self).__init__()
        in_dim = args.emb_dim
        self.gnn_dims = [in_dim,args.gat_feat_dim]

        self.gnn_layers = 1
        self.att_heads = [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GATdecoderlayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], 0.1)
            )

    def forward(self, doc_sents_h, doc_len, adj,W):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, W,adj)

        return doc_sents_h


