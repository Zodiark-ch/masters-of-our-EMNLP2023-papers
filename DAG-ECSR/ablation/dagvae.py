from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch .nn.functional as F
from ablation.gnn_utils import *

class CAVAE_dagnn(nn.Module):
    def __init__(self,args):
        super(CAVAE_dagnn,self).__init__()
        self.args=args
        self.encoder=DAGNNencoder(args)
        self.decoder=DAGNNdecoder(args)
        
    
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths):
        fU,adjB_list=self.encoder(features,adj,s_mask,s_mask_onehot,lengths)
        X,b_inv=self.decoder(fU,adj,s_mask,s_mask_onehot,lengths,adjB_list)

        return X,b_inv
        
class DAGNNencoder(nn.Module):
    def __init__(self,args):
        super(DAGNNencoder,self).__init__()
        self.args=args
        self.dropout=nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.emb_dim, args.gnn_hidden_dim)
        self.gnn_layers=args.gnn_layers
        gats=[]
        for _ in range(args.gnn_layers):
            gats += [GAT_encoder(args.gnn_hidden_dim)]
        self.gather = nn.ModuleList(gats)
        
        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)# GRUcell

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)#GRUcell
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.gnn_hidden_dim * 2, args.gnn_hidden_dim)]
        self.fcs = nn.ModuleList(fcs)
        
        self.nodal_att_type = args.nodal_att_type
        in_dim = args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        self.attentive_node_features = attentive_node_features(in_dim)
        
        # output mlp layers
        layers = [nn.Linear(args.gnn_hidden_dim * (args.gnn_layers + 1), args.gnn_hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.gnn_hidden_dim, args.gnn_hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.gnn_hidden_dim, args.code_dim)]

        self.out_mlp = nn.Sequential(*layers)
        self.largefc=nn.Linear(self.args.code_dim,args.emb_dim)
        
        

        
        
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        # if features.size()[-1]==self.args.code_dim:
        #     input=F.relu(self.largefc(features))
        # else:
        #     input=features
        num_utter = features.size()[1]
        H0 = F.relu(self.fc1(features))#(B,N,D) to (B,N,H) 
        # H0 = self.dropout(H0)
        H = [H0]### 
        adjB=torch.zeros([features.size()[0],1,num_utter]).cuda()
        adjB_list=[adjB]
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) #H[0][:,0,:]
            
            M = torch.zeros_like(C).squeeze(1) # M（B，H） 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            H1 = C+P#C and P (B,1,D)
            adjB=torch.zeros([features.size()[0],1,num_utter]).cuda()
            for i in range(1, num_utter):#
                # print(i,num_utter)
                B, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i])
                B=nn.ZeroPad2d(padding=(0,num_utter-B.size()[-1],0,0))(B.squeeze(1)).unsqueeze(1)
                
                
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                
                H_temp = C+P
                H1 = torch.cat((H1 , H_temp), dim = 1)  
                if i==0:
                    
                    adjB=B
                else:
                    adjB=torch.cat((adjB,B),dim=1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)# H=[tensor[H],tensor[H1],……]
            adjB_list.append(adjB)
        # H.append(features)#D=1024
        H = torch.cat(H, dim = 2) #(B,N,D) D=1024

        H = self.attentive_node_features(H,lengths,self.nodal_att_type) #(B,N,D) D=2224
        
        logits = self.out_mlp(H)
        
        
        return logits,adjB_list



class DAGNNdecoder(nn.Module):
    def __init__(self,args):
        super(DAGNNdecoder,self).__init__()
        self.args=args
        self.dropout=nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.code_dim, args.gnn_hidden_dim)
        self.gnn_layers=args.gnn_layers
        gats=[]
        for _ in range(args.gnn_layers):
            gats += [GNN_decoder(args.gnn_hidden_dim)]
        self.gather = nn.ModuleList(gats)
        
        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)# GRUcell

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)# GRUcell
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.gnn_hidden_dim * 2, args.gnn_hidden_dim)]
        self.fcs = nn.ModuleList(fcs)
        
        self.nodal_att_type = args.nodal_att_type
        in_dim = args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        self.attentive_node_features = attentive_node_features(in_dim)
        
        # output mlp layers
        layers = [nn.Linear(args.gnn_hidden_dim * (args.gnn_layers + 1), args.gnn_hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.gnn_hidden_dim, args.gnn_hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.gnn_hidden_dim, args.feat_dim)]

        self.out_mlp = nn.Sequential(*layers)
        
        

        


        
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths,adjB_list):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        
        num_utter = features.size()[1]
        I=torch.eye(num_utter).repeat(features.size()[0],1,1).cuda()#
        H0 = F.relu(self.fc1(features))#(B,N,D) to (B,N,H) 
        # H0 = self.dropout(H0)
        H = [H0]### 
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) #
            #
            M = torch.zeros_like(C).squeeze(1) # M（B，H） 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            H1 = C+P#C and P (B,1,D)
        
            b_inv= torch.linalg.solve(I, (I-adjB_list[l+1]))
            
            
            for i in range(1, num_utter):#
                # print(i,num_utter)
                #
                
                M = self.gather[l](H[l][:,i,:], H1, H1, b_inv[:,i,:i], s_mask[:,i,:i])
                
                #
                
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                
                H_temp = C+P
                if i==0:
                    H1=H_temp
                    
                else:
                    H1 = torch.cat((H1 , H_temp), dim = 1)  #
                    
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)# H=[tensor[H],tensor[H1],……]
            
        # H.append(features)#D=1024
        H = torch.cat(H, dim = 2) #(B,N,D) D=1024

        H = self.attentive_node_features(H,lengths,self.nodal_att_type) #(B,N,D) D=2224
        
        logits = self.out_mlp(H)
        feature_map= torch.linalg.solve(I, (I-adjB_list[2]))
        
        
        return logits,feature_map


