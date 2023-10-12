
import torch
import torch.nn as nn
import torch .nn.functional as F

from ablation.rank import *
from ablation.dagvae import*
from transformers import RobertaModel
class MyModel(nn.Module):
    def __init__(self,args):
        super(MyModel,self).__init__()
        self.args=args
        #self.bert=RobertaModel.from_pretrained(self.args.roberta_pretrain_path)
        
        self.vae=CAVAE_dagnn(args)
        self.pred1=Pre_Predictions(args)
        self.pred2=Pre_Predictions(args)
        self.rank=RankNN(args)
        self.pairwise_loss = args.pairwise_loss
        
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths,padd_adj,bert_token_b,bert_masks_b,bert_clause_b):
        if self.args.withbert==True:
        
            bert_output=self.bert(input_ids=bert_token_b.cuda(),attention_mask=bert_masks_b.cuda())
            doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.cuda())
            
            H=self.vae(doc_sents_h,adj,s_mask,s_mask_onehot,lengths)  #dagnn
            
        else:
            doc_sents_h=features
            if self.args.withvae==False:
                H,adj_map2=self.gnn(doc_sents_h,adj,s_mask,s_mask_onehot,lengths)  #dagnn
                e=0
                s=0
            else:
                H,adj_map2,e,s=self.vae(doc_sents_h,adj,s_mask,s_mask_onehot,lengths,padd_adj)  #dagnn
        #print(H.size())
        
        # pred1_e, pred1_c = self.pred1(doc_sents_h)
        pred2_e, pred2_c = self.pred2(H)
        couples_pred, emo_cau_pos = self.rank(H)
        return couples_pred, emo_cau_pos, pred2_e, pred2_c,adj_map2,e,s

    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h
    
    def loss_pre(self,pred2_e, pred2_c, y_emotions, y_causes, y_mask):
        y_mask = torch.BoolTensor(y_mask).cuda()
        y_emotions = torch.FloatTensor(y_emotions).cuda()
        y_causes = torch.FloatTensor(y_causes).cuda()

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        #pred1_e = pred1_e.masked_select(y_mask)
        pred2_e = pred2_e.masked_select(y_mask)
        true_e = y_emotions.masked_select(y_mask)
        #loss_e1 = criterion(pred1_e, true_e)
        loss_e2 = criterion(pred2_e, true_e)

        #pred1_c = pred1_c.masked_select(y_mask)
        pred2_c = pred2_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        #loss_c1 = criterion(pred1_c, true_c)
        loss_c2 = criterion(pred2_c, true_c)
        return loss_e2, loss_c2
    
    def loss_vae(self,pred1_e, pred1_c,pred2_e, pred2_c):
        # criterion_type=nn.CrossEntropyLoss(ignore_index=-1)
        # loss_type_in= criterion_type(emotion_type_in.permute(0,2,1), batch_doc_emotion_category)
        # loss_type_out=criterion_type(emotion_type_out.permute(0,2,1), batch_doc_emotion_category)
        # loss_type=loss_type_in
        
        loss_KLe=F.kl_div(pred2_e.squeeze(-1).softmax(dim=-1).log(),pred1_e.squeeze(-1).softmax(dim=-1),reduction='mean')
        loss_KLc=F.kl_div(pred2_c.squeeze(-1).softmax(dim=-1).log(),pred1_c.squeeze(-1).softmax(dim=-1),reduction='mean')
        return loss_KLe+loss_KLc
    
    def loss_KL(self,e,s):
        batch=e[1].size()[0]
        utt=e[1].size()[1]
        num=batch*utt*utt
        sum=0
        for i in range(1,self.args.gnn_layers+1):
            KLD= -0.5 * torch.sum(1 + s[i] - e[i].pow(2) - s[i].exp())
            KLD=KLD/num
            sum+=KLD
            
        # KLD= -0.5 * torch.sum(1 + s - e.pow(2) - s.exp())
        # sum=KLD
        return sum
    
    ######loss rank##### for the pairs
    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        couples_true, couples_mask, doc_couples_pred = \
        self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask, test)

        if not self.pairwise_loss:
            couples_mask = torch.BoolTensor(couples_mask).cuda()
            couples_true = torch.FloatTensor(couples_true).cuda()
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            couples_true = couples_true.masked_select(couples_mask)
            couples_pred = couples_pred.masked_select(couples_mask)
            loss_couple = criterion(couples_pred, couples_true)
        else:
            x1, x2, y = self.pairwise_util(couples_pred, couples_true, couples_mask)
            criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
            loss_couple = criterion(F.tanh(x1), F.tanh(x2), y)

        return loss_couple, doc_couples_pred
    
    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()

        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            # if test:
            K=min(20,couples_pred_i.size()[0])
            if torch.sum(torch.isnan(couples_pred_i)) > 0:
                k_idx = [0] * K
            else:
                _, k_idx = torch.topk(couples_pred_i, k=K, dim=0)
            doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred
    
    def pairwise_util(self, couples_pred, couples_true, couples_mask):
        """
        TODO: efficient re-implementation; combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()
        x1, x2 = [], []
        for i in range(batch):
            x1_i_tmp = []
            x2_i_tmp = []
            couples_mask_i = couples_mask[i]
            couples_pred_i = couples_pred[i]
            couples_true_i = couples_true[i]
            for pred_ij, true_ij, mask_ij in zip(couples_pred_i, couples_true_i, couples_mask_i):
                if mask_ij == 1:
                    if true_ij == 1:
                        x1_i_tmp.append(pred_ij.reshape(-1, 1))
                    else:
                        x2_i_tmp.append(pred_ij.reshape(-1))
            m = len(x2_i_tmp)
            n = len(x1_i_tmp)
            x1_i = torch.cat([torch.cat(x1_i_tmp, dim=0)] * m, dim=1).reshape(-1)
            x1.append(x1_i)
            x2_i = []
            for _ in range(n):
                x2_i.extend(x2_i_tmp)
            x2_i = torch.cat(x2_i, dim=0)
            x2.append(x2_i)

        x1 = torch.cat(x1, dim=0)
        x2 = torch.cat(x2, dim=0)
        y = torch.FloatTensor([1] * x1.size(0)).cuda()
        return x1, x2, y
    #######loss rank #### for the pairs    
        

    
class Pre_Predictions(nn.Module):
    def __init__(self, args):
        super(Pre_Predictions, self).__init__()
        #self.feat_dim = int(args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim)
        self.feat_dim=args.gnn_hidden_dim
        self.out_e = nn.Linear(self.feat_dim, 1)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)
        
        