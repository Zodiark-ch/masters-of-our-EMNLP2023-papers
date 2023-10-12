import sys,os,warnings,time,argparse,random
from tkinter import Y

from sympy import im
warnings.filterwarnings("ignore")
import numpy as np
import torch
import logging
from data_loader import *
import torch.nn as nn
from transformers import AdamW,get_linear_schedule_with_warmup
from model import MyModel
from train_test import train_eval
from tensorboardX import SummaryWriter
import wandb




parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='dailydialog', type= str, help='dataset name, only dailydialog')
parser.add_argument('--roberta_pretrain_path', default='./roberta_base', \
                        type= str, help='pretrain model path')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--lr', type=float, default=9e-4, metavar='LR', help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')
parser.add_argument('--epoch', type=int, default=50, metavar='E', help='number of epochs')
parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
parser.add_argument('--emb_dim', type=int, default=768, help='Feature size.')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='how many batchiszes to bp.')
parser.add_argument('--warmup_proportion', type=float, default=0.06, help='the lr up phase in the warmup.')
parser.add_argument('--windowp', type=int, default=1,
                        help='context window size for constructing edges in graph model for past utterances')
parser.add_argument('--withbert', default=False, help='')
parser.add_argument('--withvae', default=True, help='')
parser.add_argument('--iemocaptest', default=False, help='')
parser.add_argument('--try_method', default='DAGNN_RB_RANK_lr2e-5', type= str, help='tensorboardX data file name')
parser.add_argument('--gpu', type=str, default='0',  help='id of gpus')
parser.add_argument('--earlystop', type=int, default=60,  help='id of gpus')
parser.add_argument('--emotion_classes', type=int, default=7, help='neutral, anger, disgust, fear, happiness, sadness, surprise')
############model parameters####################
parser.add_argument('--gnn_layers', type=int, default=3, help='Number of gnn layers.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--gnn_hidden_dim', type = int, default=300)
parser.add_argument('--gat_feat_dim', type = int, default=192)#
parser.add_argument('--feat_dim', type = int, default=768)
parser.add_argument('--code_dim', type = int, default=192)#
parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'], help='type of nodal attention')
parser.add_argument('--rank_K', type=int, default=12, help='')
parser.add_argument('--rank_pos_emb_dim', type=int, default=50, help='')
parser.add_argument('--pairwise_loss', action='store_true', default=False, help='')
args = parser.parse_args()
if args.withbert==False:
    args.emb_dim=1024
    args.feat_dim=1024
    args.gat_feat_dim=256
    args.batch_size=32
print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
#torch.autograd.set_detect_anomaly(True)

def seed_everything(seed=args.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(seed)
seed_everything()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

logger = get_logger('saved_models/' + args.dataset_name +'_'+str(args.lr)+'_'+str(args.epoch)+ '_logging.log')
logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
logger.info(args)

cuda=torch.cuda.is_available()


def main(fold_id):

    writer=SummaryWriter('runs/'+args.try_method+'fold_{}'.format(fold_id))
    train_loader=build_train_data(fold_id=fold_id,batch_size=args.batch_size,data_type='train',args=args)
    valid_loader = build_inference_data( fold_id=fold_id,batch_size=args.batch_size,data_type='valid',args=args)
    test_loader = build_inference_data( fold_id=fold_id,batch_size=args.batch_size,data_type='test',args=args)
    model=MyModel(args)
    
    
    if cuda:
        model.cuda()

    # params = model.parameters()
    # params_bert = model.bert.parameters()
    # params_rest = list(model.vae.parameters()) + list(model.pred1.parameters()) +list(model.pred2.parameters())+ list(model.rank.parameters())
    # assert sum([param.nelement() for param in params]) == \
    #        sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])

    # no_decay = ['bias', 'LayerNorm.weight']
    # params = [
    #     {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01, 'eps': 1e-8},
    #     {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0, 'eps': 1e-8},
    #     {'params': params_rest,
    #      'weight_decay': 1e-5}
    # ]

    optimizer = AdamW(model.parameters(),lr=args.lr)
    num_steps_all = len(train_loader) // args.gradient_accumulation_steps * args.epoch #
    warmup_steps = int(num_steps_all * args.warmup_proportion)#
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)
    model.zero_grad()
    print('Data and model load finished')
    
    
    max_ec_p, max_ec_n, max_ec_avg, max_e, max_c = (-1, -1, -1), (-1, -1, -1),(-1, -1, -1),None, None
    metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = (-1, -1, -1),(-1, -1, -1),(-1, -1, -1), None, None
    
    for epoch in range(1,int(args.epoch)+1):
        
        train_eval(model,train_loader, fold_id,epoch,args,optimizer,scheduler,logger,writer,train=True)
        
        valid_ec_p, valid_ec_n, valid_ec_avg, valid_e, valid_c, doc_id_all, doc_couples_all, doc_couples_pred_all= \
        train_eval(model,valid_loader, fold_id,epoch,args,optimizer,scheduler,logger,writer,train=False)
        logger.info('VALID#: fold: {} epoch: {}, valid_ECP_Positive: {}, valid_ECP_Negative: {}, valid_ECP_average: {} \n'. \
            format(fold_id,   epoch,      valid_ec_p,             valid_ec_n,             valid_ec_avg))
        writer.add_scalars('valid metric',{'valid_ECP_Positive':valid_ec_p[2],'valid_ECP_Negative':valid_ec_n[2],'valid_ECP_average':valid_ec_avg[2]},epoch)
        test_ec_p, test_ec_n, test_ec_avg, test_e, test_c, doc_id_all, doc_couples_all, doc_couples_pred_all= \
        train_eval(model,test_loader, fold_id,epoch,args,optimizer,scheduler,logger,writer,train=False)
        logger.info('TEST#: fold: {} epoch: {}, test_ECP_Positive: {}, test_ECP_Negative: {}, test_ECP_average: {} \n'. \
            format(fold_id,   epoch,      test_ec_p,             test_ec_n,             test_ec_avg))
        writer.add_scalars('test metric',{'test_ECP_Positive':test_ec_p[2],'test_ECP_Negative':test_ec_n[2],'test_ECP_average':test_ec_avg[2]},epoch)
        print('fold:{}  epoch:{}      valid_ec:{},     test_ec:{}'.format(fold_id, epoch, valid_ec_p[2],test_ec_p[2]))
        
        if args.iemocaptest == True:
            if test_ec_p[2] > max_ec_p[2]:
                    early_stop_flag = 1
                    max_ec_p, max_ec_n, max_ec_avg, max_e, max_c = test_ec_p, test_ec_n, test_ec_avg, test_e, test_c
                    metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = test_ec_p, test_ec_n, test_ec_avg, test_e, test_c
            else:
                early_stop_flag += 1
        else:
            if valid_ec_p[2] > max_ec_p[2]:
                        early_stop_flag = 1
                        max_ec_p, max_ec_n, max_ec_avg, max_e, max_c = valid_ec_p, valid_ec_n, valid_ec_avg, valid_e, valid_c
                        metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = test_ec_p, test_ec_n, test_ec_avg, test_e, test_c
            else:
                early_stop_flag += 1
            
        if epoch > args.epoch / 2 and early_stop_flag >= args.earlystop:
            break
    
    return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c
    
        
    
    
    
    
    
    
if __name__ == '__main__':
    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    # for fold_id in range(1, n_folds+1):
    metric_ec_p_all, metric_ec_n_all, metric_ec_avg_all, metric_e_all, metric_c_all=[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]
    for fold_id in range(1,11):
        wandb_config=dict(lr=args.lr,windowp=args.windowp,batch_size=args.batch_size,fold=fold_id)
        wandb.init(config=wandb_config,reinit=True,project='ecsr_dagnn_em_vae_1',name='adj-1_lr_{}_windowp_{}_batch_{}_fold_{}'.format(args.lr,args.windowp,args.batch_size,fold_id))
        print('===== fold {} ====='.format(fold_id))
        metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = main( fold_id)
        print('F_ecp_pos: {}, P_ecp_pos: {}, R_ecp_pos: {}'.format(float_n(metric_ec_p[2]), float_n(metric_ec_p[0]), float_n(metric_ec_p[1])))
        print('F_ecp_neg: {}, P_ecp_neg: {}, R_ecp_neg: {}'.format(float_n(metric_ec_n[2]), float_n(metric_ec_n[0]), float_n(metric_ec_n[1])))
        print('F_ecp_avg: {}, P_ecp_avg: {}, R_ecp_avg: {}'.format(float_n(metric_ec_avg[2]), float_n(metric_ec_avg[0]), float_n(metric_ec_avg[1])))
        print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
        print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
        metric_ec_p_all[0]+=metric_ec_p[0] 
        metric_ec_p_all[1]+=metric_ec_p[1] 
        metric_ec_p_all[2]+=metric_ec_p[2]
        metric_ec_n_all[0]+=metric_ec_n[0]
        metric_ec_n_all[1]+=metric_ec_n[1]
        metric_ec_n_all[2]+=metric_ec_n[2]
        metric_ec_avg_all[0]+=metric_ec_avg[0]
        metric_ec_avg_all[1]+=metric_ec_avg[1]
        metric_ec_avg_all[2]+=metric_ec_avg[2]
        metric_e_all[0]+=metric_e[0]
        metric_e_all[1]+=metric_e[1]
        metric_e_all[2]+=metric_e[2]
        metric_c_all[0]+=metric_c[0]
        metric_c_all[1]+=metric_c[1]
        metric_c_all[2]+=metric_c[2]
        wandb.log({'F_ecp_pos': metric_ec_p[2],  'P_ecp_pos': metric_ec_p[0],'R_ecp_pos':metric_ec_p[1]})
        wandb.log({'F_ecp_neg': metric_ec_n[2],  'P_ecp_neg': metric_ec_n[0],'R_ecp_neg':metric_ec_n[1]})
        wandb.log({'F_ecp_avg': metric_ec_avg[2],  'P_ecp_avg': metric_ec_avg[0],'R_ecp_avg':metric_ec_avg[1]})
        wandb.log({'F_emo': metric_e[2],  'P_emo': metric_e[0],'R_emo':metric_e[1]})
        wandb.log({'F_cau': metric_c[2],  'P_cau': metric_c[0],'R_cau':metric_c[1]})
        wandb.join()
    print('======== all ========')
    print('F_ecp_pos: {}, P_ecp_pos: {}, R_ecp_pos: {}'.format(float_n(metric_ec_p_all[2]), float_n(metric_ec_p_all[0]), float_n(metric_ec_p_all[1])))
    print('F_ecp_neg: {}, P_ecp_neg: {}, R_ecp_neg: {}'.format(float_n(metric_ec_n_all[2]), float_n(metric_ec_n_all[0]), float_n(metric_ec_n_all[1])))
    print('F_ecp_avg: {}, P_ecp_avg: {}, R_ecp_avg: {}'.format(float_n(metric_ec_avg_all[2]), float_n(metric_ec_avg_all[0]), float_n(metric_ec_avg_all[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e_all[2]), float_n(metric_e_all[0]), float_n(metric_e_all[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c_all[2]), float_n(metric_c_all[0]), float_n(metric_c_all[1])))
    file=open('saved_models/'+str(args.lr)+str(args.withvae)+'.txt','w')
    results='F_ecp_pos: {}, P_ecp_pos: {}, R_ecp_pos: {},F_ecp_neg: {}, P_ecp_neg: {}, R_ecp_neg: {},F_ecp_avg: {}, P_ecp_avg: {}, R_ecp_avg: {},F_emo: {}, P_emo: {}, R_emo: {},F_cau: {}, P_cau: {}, R_cau: {}'.format( \
        float_n(metric_ec_p_all[2]), float_n(metric_ec_p_all[0]), float_n(metric_ec_p_all[1]),float_n(metric_ec_n_all[2]), float_n(metric_ec_n_all[0]), float_n(metric_ec_n_all[1]), \
            float_n(metric_ec_avg_all[2]), float_n(metric_ec_avg_all[0]), float_n(metric_ec_avg_all[1]),float_n(metric_e_all[2]), float_n(metric_e_all[0]), float_n(metric_e_all[1]),float_n(metric_c_all[2]), float_n(metric_c_all[0]), float_n(metric_c_all[1]))
    file.write(results)
    file.close()