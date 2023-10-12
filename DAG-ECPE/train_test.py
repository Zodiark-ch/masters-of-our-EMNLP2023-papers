
from sympy import false
import torch
from utils import *
fu_feature=[]
def train_eval(model,dataloader,fold,epoch,args,optimizer,scheduler,logger,writer,train=False):
    assert not model or dataloader or optimizer or scheduler!= None
    if train:
        model.train()
        logger.info('########################Training######################')
        # dataloader = tqdm(dataloader)
    else:
        model.eval()
        logger.info('########################Evaling######################')
        

    doc_id_all,doc_couples_all,doc_couples_pred_all=[],[],[]   
    y_causes_b_all = []
    
     
    for train_step, batch in enumerate(dataloader, 1):
        batch_ids,batch_doc_len,batch_pairs,label_emotions,label_causes,batch_doc_speaker,features,adj,s_mask, \
            s_mask_onehot,batch_doc_emotion_category,batch_doc_emotion_token,batch_utterances,batch_utterances_mask,batch_uu_mask, \
                bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b=batch
        
        features = features.cuda()
        adj = adj.cuda()
        s_mask = s_mask.cuda()
        s_mask_onehot = s_mask_onehot.cuda()
        batch_doc_len = batch_doc_len.cuda()
        batch_doc_emotion_category=batch_doc_emotion_category.cuda()
        
        couples_pred, emo_cau_pos, pred1_e, pred1_c,pred2_e, pred2_c,adj_map = model(features,adj,s_mask,s_mask_onehot,batch_doc_len,batch_uu_mask, \
            bert_token_b,bert_masks_b,bert_clause_b)
        
        
        
        
         
         
        loss_e, loss_c = model.loss_pre(pred1_e, pred1_c,pred2_e, pred2_c, label_emotions, label_causes, batch_utterances_mask)
        loss_couple, doc_couples_pred = model.loss_rank(couples_pred, emo_cau_pos, batch_pairs, batch_utterances_mask)
        loss_KL=model.loss_vae(pred1_e, pred1_c,pred2_e, pred2_c)
        if len(dataloader)==47:
            logger.info('VALID# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL))
            writer.add_scalars('valid_loss',{'loss_e':loss_e,'loss_c':loss_c,'loss_couple':loss_couple, \
                                              'loss_KL':loss_KL},train_step+len(dataloader)*epoch)
        if len(dataloader)==257:
            logger.info('TEST# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL))
            writer.add_scalars('test_loss',{'loss_e':loss_e,'loss_c':loss_c,'loss_couple':loss_couple, \
                                             'loss_KL':loss_KL},train_step+len(dataloader)*epoch)
        if len(dataloader)==16:
            logger.info('TEST# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL))
            writer.add_scalars('test_loss',{'loss_e':loss_e,'loss_c':loss_c,'loss_couple':loss_couple, \
                                             'loss_KL':loss_KL},train_step+len(dataloader)*epoch)
        
        
  
        if train:
            logger.info('TRAIN# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL))
            writer.add_scalars('train_loss',{'loss_e':loss_e,'loss_c':loss_c,'loss_couple':loss_couple, \
                                                 'loss_KL':loss_KL},train_step+len(dataloader)*epoch)
            loss = loss_couple + loss_e + loss_c+loss_KL
            loss = loss / args.gradient_accumulation_steps
            loss.backward()#
            if train_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
        doc_id_all.extend(batch_ids)
        doc_couples_all.extend(batch_pairs)
        doc_couples_pred_all.extend(doc_couples_pred)
        y_causes_b_all.extend(list(label_causes))
    if train==False:

        doc_couples_pred_all = lexicon_based_extraction(doc_id_all, doc_couples_pred_all,args,doc_couples_all,fold=fold)
        metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = eval_func(doc_couples_all, \
            doc_couples_pred_all, y_causes_b_all)
        return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all
        
def lexicon_based_extraction(doc_ids, couples_pred,args,doc_couples_all,fold):
    if args.iemocaptest==True and len(doc_ids)==16:
        emotional_clauses=[]
        for i in range(len(doc_ids)):
            if len(doc_couples_all[i])>0:
                emotional_clause,_=zip(*doc_couples_all[i])
            else:
                emotional_clause=[]
            emotional_clauses.append(emotional_clause)
    else:
        emotional_clauses = read_b('data/dailydialog/fold%s/sentimental_clauses.pkl'%(fold))#

    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]
        if args.iemocaptest==True and len(doc_ids)==16:
            emotional_clauses_i = emotional_clauses[i]
        else:
            emotional_clauses_i = emotional_clauses[doc_id]
        for couple in couples_pred_i[1:]:
            if couple[0][0] in emotional_clauses_i and logistic(couple[1]) > 0.5:
                couples_pred_i_filtered.append(couple[0])

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered
                
        
        