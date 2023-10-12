
from sympy import false
import torch
from utils import *
import wandb

def train_eval(model,dataloader,fold,epoch,args,optimizer,scheduler,logger,writer,train=False):
    assert not model or dataloader or optimizer or scheduler!= None
    if train:
        model.train()
        logger.info('########################Training######################')
        # dataloader = tqdm(dataloader)
    else:
        model.eval()
        logger.info('########################Evaling######################')
        

    doc_id_all,doc_couples_all,doc_couples_pred_all1,doc_couples_pred_all2,doc_couples_pred_all3,doc_couples_pred_all4,doc_couples_pred_all5,doc_couples_pred_all6,doc_label_all=[],[],[],[],[],[],[],[],[]  
    doc_couples_all1,doc_couples_all2,doc_couples_all3,doc_couples_all4,doc_couples_all5,doc_couples_all6=[],[],[],[],[],[]
    y_causes_b_all1,y_causes_b_all2,y_causes_b_all3,y_causes_b_all4,y_causes_b_all5,y_causes_b_all6 = [],[],[],[],[],[]
    
     
    for train_step, batch in enumerate(dataloader, 1):
        batch_ids,batch_doc_len,batch_pairs,label_emotions,label_causes,batch_doc_speaker,features,adj,s_mask, \
            s_mask_onehot,batch_doc_emotion_category,batch_doc_emotion_token,batch_utterances,batch_utterances_mask,batch_uu_mask, \
                bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b,batch_doc_label_list=batch
        
        features = features.cuda()
        adj = adj.cuda()
        s_mask = s_mask.cuda()
        s_mask_onehot = s_mask_onehot.cuda()
        batch_doc_len = batch_doc_len.cuda()
        batch_doc_emotion_category=batch_doc_emotion_category.cuda()
        batch_doc_label_list
        
        batch_pair1_all,batch_pair2_all,batch_pair3_all,batch_pair4_all,batch_pair5_all,batch_pair6_all=[],[],[],[],[],[]
        
        label_causes1=label_causes
        label_causes2=label_causes
        label_causes3=label_causes
        label_causes4=label_causes
        label_causes5=label_causes
        
        for i in range(len(batch_ids)):
            batch_pair1,batch_pair2,batch_pair3,batch_pair4,batch_pair5,batch_pair6=[],[],[],[],[],[]
            
            for typei in range(5):
                
                for j in range(len(batch_pairs[i])):
                    if batch_doc_label_list[i][j]==typei and typei==0:
                        batch_pair1.append(batch_pairs[i][j])
                    if batch_doc_label_list[i][j]==typei and typei==1:
                        batch_pair2.append(batch_pairs[i][j])
                    if batch_doc_label_list[i][j]==typei and typei==2:
                        batch_pair3.append(batch_pairs[i][j])
                    if batch_doc_label_list[i][j]==typei and typei==3:
                        batch_pair4.append(batch_pairs[i][j])
                    if batch_doc_label_list[i][j]==typei and typei==4:
                        batch_pair5.append(batch_pairs[i][j])
            if len(batch_pair1)>0:
                emotion,cause=zip(*batch_pair1)
            else:
                cause=[]
            for j in range(len(label_causes1[i])):
                if j+1 not in cause:
                    label_causes1[i][j]=0
            
            if len(batch_pair2)>0:
                emotion,cause=zip(*batch_pair2)
            else:
                cause=[]
            for j in range(len(label_causes2[i])):
                if j+1 not in cause:
                    label_causes2[i][j]=0
            
            if len(batch_pair3)>0:
                emotion,cause=zip(*batch_pair3)
            else:
                cause=[]
            for j in range(len(label_causes3[i])):
                if j+1 not in cause:
                    label_causes3[i][j]=0
                    
            if len(batch_pair4)>0:
                emotion,cause=zip(*batch_pair4)
            else:
                cause=[]
            for j in range(len(label_causes4[i])):
                if j+1 not in cause:
                    label_causes4[i][j]=0
                    
            if len(batch_pair5)>0:
                emotion,cause=zip(*batch_pair5)
            else:
                cause=[]
            for j in range(len(label_causes5[i])):
                if j+1 not in cause:
                    label_causes5[i][j]=0
                    
           
                    
            batch_pair1_all.append(batch_pair1)
            batch_pair2_all.append(batch_pair2)
            batch_pair3_all.append(batch_pair3)
            batch_pair4_all.append(batch_pair4)
            batch_pair5_all.append(batch_pair5)
            
        
              
        doc_id_all.extend(batch_ids)
        doc_couples_all1.extend(batch_pair1_all)
        doc_couples_all2.extend(batch_pair2_all)
        doc_couples_all3.extend(batch_pair3_all)
        doc_couples_all4.extend(batch_pair4_all)
        doc_couples_all5.extend(batch_pair5_all)
        
        
        
        couples_pred, emo_cau_pos,pred2_e, pred2_c,adj_map2,e,s = model(features,adj,s_mask,s_mask_onehot,batch_doc_len,batch_uu_mask, \
            bert_token_b,bert_masks_b,bert_clause_b)
        couples_pred1=couples_pred[:,:,0]
        couples_pred2=couples_pred[:,:,1]
        couples_pred3=couples_pred[:,:,2]
        couples_pred4=couples_pred[:,:,3]
        
        
        
        loss_couple1, doc_couples_pred1 = model.loss_rank(couples_pred1, emo_cau_pos, batch_pair1_all, batch_utterances_mask)
        loss_couple2, doc_couples_pred2 = model.loss_rank(couples_pred2, emo_cau_pos, batch_pair2_all, batch_utterances_mask)
        loss_couple3, doc_couples_pred3 = model.loss_rank(couples_pred3, emo_cau_pos, batch_pair3_all, batch_utterances_mask)
        loss_couple4, doc_couples_pred4 = model.loss_rank(couples_pred4, emo_cau_pos, batch_pair4_all, batch_utterances_mask)
        #loss_couple5, doc_couples_pred5 = model.loss_rank(couples_pred, emo_cau_pos, batch_pair5_all, batch_utterances_mask)
        loss_couple=loss_couple1+loss_couple2+loss_couple3+loss_couple4
        loss_couple=loss_couple/4
        
         
         
        loss_e, loss_c = model.loss_pre(pred2_e, pred2_c, label_emotions, label_causes, batch_utterances_mask)
        loss_KL=model.loss_KL(e,s)
        #loss_couple, doc_couples_pred = model.loss_rank(couples_pred, emo_cau_pos, batch_pairs, batch_utterances_mask)
        #loss_KL=model.loss_vae(pred1_e, pred1_c,pred2_e, pred2_c)
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
            wandb.log({'epoch': epoch,  'step':train_step+len(dataloader)*epoch,'loss_all':loss,'loss_couple':loss_couple,'loss_e':loss_e,'loss_c':loss_c,'loss_KL':loss_KL})
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if train_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        
    
        
        
        doc_label_all.extend(batch_doc_label_list)
        doc_couples_pred_all1.extend(doc_couples_pred1)
        doc_couples_pred_all2.extend(doc_couples_pred2)
        doc_couples_pred_all3.extend(doc_couples_pred3)
        doc_couples_pred_all4.extend(doc_couples_pred4)
        #doc_couples_pred_all5.extend(doc_couples_pred5)
        
        y_causes_b_all1.extend(list(label_causes1))
        y_causes_b_all2.extend(list(label_causes2))
        y_causes_b_all3.extend(list(label_causes3))
        y_causes_b_all4.extend(list(label_causes4))
        #y_causes_b_all5.extend(list(label_causes5))
        
    if train==False:

        doc_couples_pred_all1 = lexicon_based_extraction(doc_id_all, doc_couples_pred_all1,doc_couples_all1,fold=fold)
        doc_couples_pred_all2 = lexicon_based_extraction(doc_id_all, doc_couples_pred_all2,doc_couples_all2,fold=fold)
        doc_couples_pred_all3 = lexicon_based_extraction(doc_id_all, doc_couples_pred_all3,doc_couples_all3,fold=fold)
        doc_couples_pred_all4 = lexicon_based_extraction(doc_id_all, doc_couples_pred_all4,doc_couples_all4,fold=fold)
        #doc_couples_pred_all5 = lexicon_based_extraction(doc_id_all, doc_couples_pred_all5,doc_couples_all5,fold=fold)
        
        metric_ec_p1, metric_ec_n1, metric_ec_avg1, metric_e1, metric_c1 = eval_func(doc_label_all,doc_couples_all1, \
            doc_couples_pred_all1, y_causes_b_all1)
        metric_ec_p2, metric_ec_n2, metric_ec_avg2, metric_e2, metric_c2 = eval_func(doc_label_all,doc_couples_all2, \
            doc_couples_pred_all2, y_causes_b_all2)
        metric_ec_p3, metric_ec_n3, metric_ec_avg3, metric_e3, metric_c3 = eval_func(doc_label_all,doc_couples_all3, \
            doc_couples_pred_all3, y_causes_b_all3)
        metric_ec_p4, metric_ec_n4, metric_ec_avg4, metric_e4, metric_c4 = eval_func(doc_label_all,doc_couples_all4, \
            doc_couples_pred_all4, y_causes_b_all4)
        #metric_ec_p5, metric_ec_n5, metric_ec_avg5, metric_e5, metric_c5 = eval_func(doc_label_all,doc_couples_all5, \
            #doc_couples_pred_all5, y_causes_b_all5)
        
        metric_ec_p=[0,0,0]
        metric_ec_n=[0,0,0]
        metric_ec_avg=[0,0,0]
        metric_e=[0,0,0]
        metric_c=[0,0,0]
        for i in range(3):
            metric_ec_p[i]=(metric_ec_p1[i]+metric_ec_p2[i]+metric_ec_p3[i]+metric_ec_p4[i])/4
            metric_ec_n[i]=(metric_ec_n1[i]+metric_ec_n2[i]+metric_ec_n3[i]+metric_ec_n4[i])/4
            metric_ec_avg[i]=(metric_ec_avg1[i]+metric_ec_avg2[i]+metric_ec_avg3[i]+metric_ec_avg4[i])/4
            metric_e[i]=(metric_e1[i]+metric_e2[i]+metric_e3[i]+metric_e4[i])/4
            metric_c[i]=(metric_c1[i]+metric_c2[i]+metric_c3[i]+metric_c4[i])/4
            
        #print(metric_ec_p1[2],metric_ec_p2[2],metric_ec_p3[2],metric_ec_p4[2],metric_ec_p5[2])
        return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all1
        
def lexicon_based_extraction(doc_ids, couples_pred,doc_couples_all,fold):
    #emotional_clauses = read_b('data/dailydialog/fold%s/sentimental_clauses.pkl'%(fold))
    emotional_clauses=[]
    for i in range(len(doc_ids)):
        if len(doc_couples_all[i])>0:
            emotional_clause,_=zip(*doc_couples_all[i])
        else:
            emotional_clause=[]
        emotional_clauses.append(emotional_clause)
    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]

        emotional_clauses_i = emotional_clauses[i]
        for couple in couples_pred_i[1:]:
            if couple[0][0] in emotional_clauses_i and couple[0][0]>=couple[0][1]:
                couples_pred_i_filtered.append(couple[0])

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered
                
        
        