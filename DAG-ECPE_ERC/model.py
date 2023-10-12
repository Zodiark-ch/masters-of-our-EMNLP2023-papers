import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelWithLMHead
from model_utils import *
from dagvae import *


class BertERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)
        # bert_encoder
        self.bert_config = BertConfig.from_json_file(args.bert_model_dir + 'config.json')

        self.bert = BertModel.from_pretrained(args.home_dir + args.bert_model_dir, config = self.bert_config)
        in_dim =  args.bert_dim

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers- 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, content_ids, token_types,utterance_len,seq_len):

        # the embeddings for bert
        # if len(content_ids)>512:
        #     print('ll')

        #
        ## w token_type_ids
        # lastHidden = self.bert(content_ids, token_type_ids = token_types)[1] #(N , D)
        ## w/t token_type_ids
        lastHidden = self.bert(content_ids)[1] #(N , D)

        final_feature = self.dropout(lastHidden)

        # pooling

        outputs = self.out_mlp(final_feature) #(N, D)

        return outputs


class DAGERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_emb = nn.Embedding(2,args.hidden_dim)
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        else:
            gats = []
            for _ in range(args.gnn_layers):
                gats += [Gatdot(args.hidden_dim) if args.no_rel_attn else Gatdot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus = []
        for _ in range(args.gnn_layers):
            grus += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus = nn.ModuleList(grus)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, adj,s_mask):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :return:
        '''
        num_utter = features.size()[1]
        if self.rel_attn:
            rel_ft = self.rel_emb(s_mask) # (B, N, N, D)

        H0 = F.relu(self.fc1(features)) # (B, N, D)
        H = [H0]
        for l in range(self.args.gnn_layers):
            H1 = self.grus[l](H[l][:,0,:]).unsqueeze(1) # (B, 1, D)
            for i in range(1, num_utter):
                if not self.rel_attn:
                    _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i])
                else:
                    _, M = self.gather[l](H[l][:, i, :], H1, H1, adj[:, i, :i], rel_ft[:, i, :i, :])
                H1 = torch.cat((H1 , self.grus[l](H[l][:,i,:], M).unsqueeze(1)), dim = 1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
            H0 = H1
        H.append(features)
        H = torch.cat(H, dim = 2) #(B, N, l*D)
        logits = self.out_mlp(H)
        return logits


    
def matrix_re(M):
    N=M.size()[-1]
    t=0
    for batch in range(M.size()[0]):
        for i in range(N):
            for j in range(N):
                t=M[batch][i][j]
                M[batch][i][j]=M[batch][-i][-j]
                M[batch][-i][-j]=t
    return M
        
class DAGERC_fushion(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers #default is 2

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            gats = []
            for _ in range(args.gnn_layers):
                # gats += [GAT_dialoggcn(args.hidden_dim)]
                gats += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs = nn.ModuleList(fcs)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        self.nodal_att_type = args.nodal_att_type
        
        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

    def forward(self, features, adj,s_mask,s_mask_onehot, lengths):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        ###print('H0 is:', H0.size())
        # H0 = self.dropout(H0)
        H = [H0]
        
        for l in range(self.args.gnn_layers):
            state=H[l][:,0,:]
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) 
            
            M = torch.zeros_like(C).squeeze(1) # M（B，H） 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    Q=H[l][:,i,:]
                    K=H1
                    V=H1
                    causal_adj=adj[:,i,:i]
                    speaker=s_mask[:,i,:i]
                    
                    B, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i])
                    B=nn.ZeroPad2d(padding=(0,num_utter-B.size()[-1],0,0))(B.squeeze(1))
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i])
                    else:
                        _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:, i, :i])

                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P 
                H1 = torch.cat((H1 , H_temp), dim = 1)  
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
        H.append(features)
        
        H = torch.cat(H, dim = 2) 

        H = self.attentive_node_features(H,lengths,self.nodal_att_type) 

        logits = self.out_mlp(H)

        return logits


class DAGERC_v2(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers
  
        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        else:
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        in_dim = args.hidden_dim * (args.gnn_layers * 2 + 1) + args.emb_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, adj,s_mask):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :return:
        '''
        num_utter = features.size()[1]
        if self.rel_attn:
            rel_ft = self.rel_emb(s_mask) # (B, N, N, D)

        H0 = F.relu(self.fc1(features)) # (B, N, D)
        H = [H0]
        C = [H0]
        for l in range(self.args.gnn_layers):
            CL = self.grus_c[l](C[l][:,0,:]).unsqueeze(1) # (B, 1, D)
            M = torch.zeros_like(CL).squeeze(1)
            # P = M.unsqueeze(1)
            P = self.grus_p[l](M, C[l][:,0,:]).unsqueeze(1) # (B, 1, D)
            for i in range(1, num_utter):
                if not self.rel_attn:
                    _, M = self.gather[l](C[l][:,i,:], P, P, adj[:,i,:i])
                else:
                    _, M = self.gather[l](C[l][:, i, :], P, P, adj[:, i, :i], rel_ft[:, i, :i, :])

                C_ = self.grus_c[l](C[l][:,i,:], M).unsqueeze(1)# (B, 1, D)
                P_ = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)# (B, 1, D)
                # P = M.unsqueeze(1)
                CL = torch.cat((CL, C_), dim = 1) # (B, i, D)
                P = torch.cat((P, P_), dim = 1) # (B, i, D)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            C.append(CL)
            H.append(CL)
            H.append(P)
        H.append(features)
        H = torch.cat(H, dim = 2) #(B, N, l*D)
        logits = self.out_mlp(H)
        return logits


class MyModel(nn.Module):
    
    def __init__(self, args, num_class):
        super(MyModel,self).__init__()
        self.args=args
        #self.bert=RobertaModel.from_pretrained(self.args.roberta_pretrain_path)
        if self.args.withvae==False:
      
        self.vae=CAVAE_dagnn(args)
        self.pred1=Pre_Predictions(args,num_class)
        self.pred2=Pre_Predictions(args,num_class)
    
    def forward(self, features, adj,s_mask,s_mask_onehot, lengths):
        doc_sents_h=features

        H,e,s=self.vae(doc_sents_h,adj,s_mask,s_mask_onehot,lengths)  #dagnn
        
        #pred1 = self.pred1(doc_sents_h)
        pred2 = self.pred2(H)
        return  pred2,e,s
    
    def loss_vae(self,pred1, pred2):
        # criterion_type=nn.CrossEntropyLoss(ignore_index=-1)
        # loss_type_in= criterion_type(emotion_type_in.permute(0,2,1), batch_doc_emotion_category)
        # loss_type_out=criterion_type(emotion_type_out.permute(0,2,1), batch_doc_emotion_category)
        # loss_type=loss_type_in
        
        loss_KLe=F.kl_div(pred2.squeeze(-1).softmax(dim=-1).log(),pred1.squeeze(-1).softmax(dim=-1),reduction='batchmean')
        return loss_KLe
    
    def loss_KL(self,e,s):
        batch=e[1].size()[0]
        utt=e[1].size()[1]
        num=batch*utt*utt
        sum=0
        for i in range(1,self.args.gnn_layers+1):
            KLD= -0.5 * torch.sum(1 + s[i] - e[i].pow(2) - s[i].exp())
            KLD=KLD/num
            sum+=KLD
        
        # batch=e.size()[0]
        # utt=e.size()[1]
        # num=batch*utt*utt    
        # KLD= -0.5 * torch.sum(1 + s - e.pow(2) - s.exp())
        # sum=KLD/num
        return sum
    
            
class Pre_Predictions(nn.Module):
    def __init__(self, args,num_class):
        super(Pre_Predictions, self).__init__()
        #self.feat_dim = int(args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim)
        self.feat_dim=args.gnn_hidden_dim
        self.out_e = nn.Linear(self.feat_dim, num_class)

    def forward(self, doc_sents_h):
        pred = self.out_e(doc_sents_h)
        
        return pred
    

    
    