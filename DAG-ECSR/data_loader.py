import sys
import torch
from torch.utils.data import Dataset
from utils import *
from torch.nn.utils.rnn import pad_sequence
import scipy.sparse as sp
from transformers import RobertaTokenizer


def build_train_data(fold_id, batch_size,data_type,args,shuffle=True):
    train_dataset = MyDataset(fold_id, data_type='train',args=args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               collate_fn=train_dataset.collate_fn,shuffle=shuffle)
    return train_loader

def build_inference_data(fold_id, batch_size,args,data_type):
    dataset = MyDataset( fold_id, data_type=data_type,args=args)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                               collate_fn=dataset.collate_fn,shuffle=False)
    return data_loader

class MyDataset(Dataset):
    def __init__(self,fold_id,data_type,args):
        self.data_type=data_type
        if args.iemocaptest == True and data_type=='test':
            self.data_dir='data/iemocap/iemocap_test_cls.json'
            self.speaker_vocab=pickle.load(open('data/iemocap/speaker_vocab.pkl', 'rb'))
            self.label_vocab=pickle.load(open('data/iemocap/label_vocab.pkl' , 'rb'))
        else:
            self.data_dir='data/dailydialog/fold%s/dailydialog_%s_cls.json'%(fold_id,data_type)
            self.speaker_vocab={'stoi': {'A': 0, 'B': 1}, 'itos': ['A', 'B']}
            self.label_vocab ={'stoi': {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}, 'itos': ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']}
        self.args=args
        self.bert_tokenizer = RobertaTokenizer.from_pretrained(self.args.roberta_pretrain_path,local_files_only=True)
        
        

        self.doc_id_list,self.doc_len_list,self.doc_couples_list,self.doc_speaker_list,self.doc_cls_list, \
        self.y_emotions_list,self.y_causes_list,self.doc_text_list,self.doc_emotion_category_list, \
        self.doc_emotion_token_list,self.bert_token_idx_list,self.bert_clause_idx_list, \
           self.bert_segments_idx_list,self.bert_token_lens_list,self.doc_label_list= self.read_data_file(self.data_dir)
    
    def __len__(self):
        return len(self.y_emotions_list)
    
    def __getitem__(self, idx):
        doc_id,doc_len,doc_couples,y_emotions,y_causes=self.doc_id_list[idx],self.doc_len_list[idx],self.doc_couples_list[idx],self.y_emotions_list[idx],self.y_causes_list[idx]
        doc_speaker,doc_text,doc_cls=self.doc_speaker_list[idx],self.doc_text_list[idx],self.doc_cls_list[idx]
        doc_emotion_category,doc_emotion_token=self.doc_emotion_category_list[idx],self.doc_emotion_token_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        doc_label_list=self.doc_label_list[idx]

        if bert_token_lens > 512 and self.args.withbert==True:
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category = \
                self.token_trunk(bert_token_idx, bert_clause_idx,bert_segments_idx, bert_token_lens,
                                doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)
        assert doc_len == len(y_emotions)
        return doc_id,doc_len,doc_couples,y_emotions,y_causes,doc_speaker,doc_text,doc_cls,doc_emotion_category,doc_emotion_token, \
            bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens,doc_label_list
        
    def read_data_file(self, data_dir):
        datafile=data_dir
        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        doc_speaker_list=[]
        doc_cls_list=[]
        doc_text_list=[]
        doc_label_list=[]
        doc_emotion_category_list = []
        doc_emotion_token_list=[]
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        data=read_json(datafile)
        for doc in data:
            doc_id=doc['doc_id']#
            doc_len=doc['doc_len']#
            doc_couples = doc['pairs']#
            doc_emotions, doc_causes = zip(*doc_couples)#
            doc_clauses = doc['clauses']
            doc_speaker_all=doc['speakers']
            doc_cls=doc['cls']
            doc_label=doc['label']
            doc_id_list.append(doc_id)##
            doc_len_list.append(doc_len)#
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)#
            doc_label_list.append(doc_label)
            doc_cls_list.append(doc_cls)#
            
            y_emotions, y_causes = [], []
            
            doc_str = ''
            doc_text = []
            doc_emotion_category = []
            doc_emotion_token=[]
            doc_str=''
            doc_speaker=[0]*doc_len
            for i in range(doc_len):
                emotion_label = int(i + 1 in doc_emotions)
                cause_label = int(i + 1 in doc_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)
                doc_speaker[i]=self.speaker_vocab['stoi'][doc_speaker_all[i]]               
                clause_id = doc_clauses[i]['clause_id']#
                assert int(clause_id) == i + 1
                doc_text.append(doc_clauses[i]['clause'])
                if self.args.iemocaptest == True and self.data_type=='test':
                    doc_emotion_category.append(self.label_vocab['stoi']['neu'])
                else:
                    doc_emotion_category.append(self.label_vocab['stoi']['neutral'])
                doc_emotion_token.append(doc_clauses[i]['emotion_token'])
                doc_str+='<s>'+doc_clauses[i]['clause']+' </s> '
            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)
            doc_text_list.append(doc_text)
            doc_speaker_list.append(doc_speaker)
            doc_emotion_category_list.append(doc_emotion_category)
            doc_emotion_token_list.append(doc_emotion_token)
            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)
            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 0]
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 0]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            assert len(clause_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)
            
        return doc_id_list,doc_len_list,doc_couples_list,doc_speaker_list,doc_cls_list, \
        y_emotions_list,y_causes_list,doc_text_list,doc_emotion_category_list,doc_emotion_token_list, \
            bert_token_idx_list,bert_clause_idx_list,bert_segments_idx_list,bert_token_lens_list,doc_label_list
    
    
    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category,doc_label_list):
        # TODO: cannot handle some extreme cases now
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause > doc_len / 2:
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    doc_couples = [[emotion - i, cause - i]]
                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    doc_speaker=doc_speaker[i:]
                    doc_emotion_category=doc_emotion_category[i:]
                    doc_len = doc_len - i
                    doc_label_list=doc_label_list[i:]
                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2:
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]
                    doc_speaker=doc_speaker[:i]
                    doc_emotion_category=doc_emotion_category[:i]
                    doc_label_list=doc_label_list[:i]
                    doc_len = i
                    break
                i = i - 1
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category,doc_label_list

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):             
                    a[i,j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)
    
    
    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)
    
    
    def pad_matrices(self,doc_len):
        N = max(doc_len)
        adj_b = []
        for dl in doc_len:
            adj = np.ones((dl, dl))
            adj = sp.coo_matrix(adj)
            adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                                shape=(N, N), dtype=np.float32)
            adj_b.append(adj.toarray())
        return adj_b

    def collate_fn(self,batch):
        '''
        :param data:
            doc_id,doc_len,doc_couples,y_emotions,y_causes,doc_speaker,doc_text,doc_cls,
            doc_emotion_category,doc_emotion_token
        :return:
            B:batch_size  N:batch_max_doc_len
            batch_ids:(B)
            batch_doc_len(B)
            batch_pairs(B,) not a tensor
            label_emotions:(B,N) padded
            label_causes:(B,N) padded
            batch_doc_speaker:(B,N) padded
            features: (B, N, D) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            batch_doc_emotion_category: (B, ) not a tensor
            batch_doc_emotion_token:(B, ) not a tensor
            batch_utterances:  not a tensor
            batch_utterances_mask:(B,N)
            batch_uu_mask (B,N,N)
        '''
        
        doc_id,doc_len,doc_couples,y_emotions,y_causes,doc_speaker,doc_text,doc_cls, doc_emotion_category, \
        doc_emotion_token,bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b,doc_label_list=zip(*batch)
        max_dialog_len=max(int(length) for length in doc_len)
        features=pad_sequence([torch.FloatTensor(cls) for cls in doc_cls],batch_first=True)#(B,N,D)
        label_emotions=pad_sequence([torch.LongTensor(emo) for emo in y_emotions],batch_first=True,padding_value=-1) #(B,N)
        label_causes=pad_sequence([torch.LongTensor(cau) for cau in y_causes],batch_first=True,padding_value=-1)#(B,N)
        adj=self.get_adj([sp for sp in doc_speaker],max_dialog_len)
        s_mask,s_mask_onehot=self.get_s_mask([sp for sp in doc_speaker],max_dialog_len) 
        batch_doc_speaker=pad_sequence([torch.LongTensor(speaker) for speaker in doc_speaker], batch_first=True,padding_value=-1)
        batch_ids=[docid for docid in doc_id]
        batch_doc_len=torch.LongTensor([doclen for doclen in doc_len])
        batch_pairs=[pairs for pairs in doc_couples]
        batch_doc_label_list=[label for label in doc_label_list]
        # batch_doc_emotion_category=[dec for dec in doc_emotion_category]
        batch_doc_emotion_category=pad_sequence([torch.LongTensor(dec) for dec in doc_emotion_category], batch_first=True,padding_value=-1)
        batch_doc_emotion_token=[det for det in doc_emotion_token]
        batch_utterances=[utt for utt in doc_text]
        zero=torch.zeros_like(label_emotions)
        batch_utterances_mask=torch.ones_like(label_emotions)
        batch_utterances_mask=torch.where(label_emotions==-1,zero,batch_utterances_mask)
        batch_uu_mask=self.pad_matrices(doc_len)
        bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
        bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
        bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)
        bsz, max_len = bert_token_b.size()
        bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
        for index, seq_len in enumerate(bert_token_lens_b):
            bert_masks_b[index][:seq_len] = 1

        bert_masks_b = torch.FloatTensor(bert_masks_b)
        assert bert_segment_b.shape == bert_token_b.shape
        assert bert_segment_b.shape == bert_masks_b.shape
        
        return batch_ids,batch_doc_len,batch_pairs,np.array(label_emotions),np.array(label_causes), \
            batch_doc_speaker,features,adj, \
            s_mask,s_mask_onehot,batch_doc_emotion_category,batch_doc_emotion_token,batch_utterances, \
            np.array(batch_utterances_mask),np.array(batch_uu_mask),bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b,np.array(batch_doc_label_list)
        
                       
                