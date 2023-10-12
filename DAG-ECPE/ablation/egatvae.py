from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch .nn.functional as F
from ablation.gnn_utils import *

class CAVAE_egat(nn.Module):
    def __init__(self,args):
        super(CAVAE_egat,self).__init__()
        self.args=args
        self.encoder=EGATVAEencoder(args)
        self.decoder=EGATVAEdecoder(args)
        self.fce=nn.Linear(1,1)
        self.fcs=nn.Linear(1,1)
        
    def sampling(self,mu, log_var):
        # result=[]
        # result.append(mu[0])
        # for i in range(1,self.args.gnn_layers+1):
            
        #     std = torch.exp(0.5*log_var[i])
        #     eps = torch.randn_like(std)
        #     result.append(eps.mul(std).add_(mu[i]))
            
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        result=eps.mul(std).add_(mu)
        return result # return z sample
    
    def forward(self, doc_sents_h, doc_len, adj,s_mask):
        fU,adjB_list=self.encoder(doc_sents_h,doc_len,adj,s_mask)
        adj_e=torch.clone(adjB_list).view(-1,1)
        adj_s=torch.clone(adjB_list).view(-1,1)
        e=self.fce(adj_e).view(adjB_list.size()[0],adjB_list.size()[1],adjB_list.size()[2],-1)
        s=self.fcs(adj_s).view(adjB_list.size()[0],adjB_list.size()[1],adjB_list.size()[2],-1)
        z=self.sampling(e,s)
        X=self.decoder(fU,doc_len,adj,s_mask,z)

        return X,adjB_list,e,s
        
class EGATVAEencoder(nn.Module):
    def __init__(self, args):
        super(EGATVAEencoder, self).__init__()
        in_dim = args.emb_dim
        self.gnn_dims = [in_dim,args.gat_feat_dim]

        self.gnn_layers = 1
        self.att_heads = [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                EGATencoderlayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], args.dropout)
            )

    def forward(self, doc_sents_h, doc_len, adj,s_mask):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h,W = gnn_layer(doc_sents_h, adj,s_mask)

        return doc_sents_h,W



class EGATVAEdecoder(nn.Module):
    def __init__(self, args):
        super(EGATVAEdecoder, self).__init__()
        in_dim = args.emb_dim
        self.gnn_dims = [in_dim,args.gat_feat_dim]

        self.gnn_layers = 1
        self.att_heads = [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                EGATdecoderlayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], args.dropout)
            )

    def forward(self, doc_sents_h, doc_len, adj,s_mask,W):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj,s_mask,W)

        return doc_sents_h


