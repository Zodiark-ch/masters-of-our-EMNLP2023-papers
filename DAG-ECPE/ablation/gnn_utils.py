import torch
import torch.nn as nn
import torch .nn.functional as F
import torch.nn.init as init


def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    return alpha - (1 - adj) * 1e30



class GAT_dagnn(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, hidden_size):
        super(GAT_dagnn,self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        B = K.size()[0]
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        #alpha = F.leaky_relu(alpha)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)  # (B, 1, N)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V) # (B, N, D)
        V1 = self.Wr1(V) # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()   # (B, N, 1)
        V = V0 * s_mask + V1 * (1 - s_mask)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum


class GAT_encoder(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, hidden_size):
        super(GAT_encoder,self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        B = K.size()[0]
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        #alpha = F.leaky_relu(alpha)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)  # (B, 1, N)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V) # (B, N, D)
        V1 = self.Wr1(V) # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()   # (B, N, 1)
        V = V0 * s_mask + V1 * (1 - s_mask)#V=(B,N,D)
        #PA=torch.bmm(attn_weight, V)
        attn_sum = Q[:,0,:].unsqueeze(1)-torch.bmm(attn_weight, V) # 
        attn_sum=attn_sum.squeeze(1)#(B,D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum    



class GNN_decoder(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, hidden_size):
        super(GNN_decoder,self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        adj = adj.unsqueeze(1)
        adj = F.softmax(adj, dim = 2).cuda() # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V) # (B, N, D)
        V1 = self.Wr1(V) # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()   # (B, N, 1)
        V = V0 * s_mask + V1 * (1 - s_mask)#V=(B,N,D)

        attn_sum = Q.unsqueeze(1)+torch.bmm(adj, V) #
        # print('attn_sum',attn_sum.size())

        return  attn_sum 



class attentive_node_features(nn.Module):
    '''
    Method to obtain attentive node features over the graph convoluted features
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(self,features, lengths, nodal_att_type):
        '''
        features : (B, N, V)
        lengths : (B, )
        nodal_att_type : type of the final nodal attention
        '''

        if nodal_att_type==None:
            return features

        batch_size = features.size(0)
        max_seq_len = features.size(1)
        padding_mask = [l*[1]+(max_seq_len-l)*[0] for l in lengths]
        padding_mask = torch.tensor(padding_mask).to(features)    # (B, N)
        causal_mask = torch.ones(max_seq_len, max_seq_len).to(features)  # (N, N)
        causal_mask = torch.tril(causal_mask).unsqueeze(0)  # (1, N, N)

        if nodal_att_type=='global':
            mask = padding_mask.unsqueeze(1)
        elif nodal_att_type=='past':
            mask = padding_mask.unsqueeze(1)*causal_mask

        x = self.transform(features)  # (B, N, V)
        temp = torch.bmm(x, features.permute(0,2,1))
        #print(temp)
        alpha = F.softmax(torch.tanh(temp), dim=2)  # (B, N, N)
        alpha_masked = alpha*mask  # (B, N, N)
        
        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # (B, N, 1)
        #print(alpha_sum)
        alpha = alpha_masked / alpha_sum    # (B, N, N)
        attn_pool = torch.bmm(alpha, features)  # (B, N, V)

        return attn_pool
    
    
    
class GraphAttentionLayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)

        attn_src = torch.matmul(F.tanh(h), self.w_src)
        attn_dst = torch.matmul(F.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        adj = torch.FloatTensor(adj).cuda()
        mask = 1 - adj.unsqueeze(1)
        # print(attn.size())
        # print(mask.size())
        attn.data.masked_fill_(mask.byte(), -999)

        attn = F.softmax(attn, dim=-1)
        feat_out = torch.matmul(attn, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    

class EGATdecoderlayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(EGATdecoderlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W_1 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_2 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_3 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.b = nn.Parameter(torch.Tensor(self.out_dim)) #(192)

        self.w_src_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        self.w_dst_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
       
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)
        
        

    def init_gnn_param(self):
        init.xavier_uniform_(self.W_1.data)
        init.xavier_uniform_(self.W_2.data)
        init.xavier_uniform_(self.W_3.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src_1.data)
        init.xavier_uniform_(self.w_dst_1.data)
       

    def forward(self, feat_in, adj, s_mask,W):
        
        
        adj = torch.FloatTensor(adj).cuda().unsqueeze(1)
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim
        I=torch.eye(N).repeat(batch,self.att_head,1,1).cuda()#
        attn,_=torch.solve(I,W)
        eye=torch.eye(N,N).cuda()
        eye=eye.unsqueeze(0).unsqueeze(0)
        eye=eye.expand(batch,-1,-1,-1)
        s_mask=s_mask.float().unsqueeze(1) #(B,1,N,N)
        inter_clause=torch.where(s_mask==1,attn,s_mask)#
        outer_clause=torch.where(s_mask==0,attn,1-s_mask)#
        intra_clause=torch.where(eye==1,adj,eye)#
        
        

        feat_in_ = feat_in.unsqueeze(1) #(B,1,N,D) 
        #h1 = torch.matmul(feat_in_, self.W_1) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h2 = torch.matmul(feat_in_, self.W_2) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h3 = torch.matmul(feat_in_, self.W_3) #(B,1,N,D)X(4,D,192)=(B,4,N,192)

        
        
        feat_out = torch.matmul(inter_clause, h2)+torch.matmul(outer_clause, h3)+ self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    
   
class EGATencoderlayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(EGATencoderlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W_1 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_2 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_3 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.b = nn.Parameter(torch.Tensor(self.out_dim)) #(192)

        self.w_src_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        self.w_dst_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
       
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)
        
        

    def init_gnn_param(self):
        init.xavier_uniform_(self.W_1.data)
        init.xavier_uniform_(self.W_2.data)
        init.xavier_uniform_(self.W_3.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src_1.data)
        init.xavier_uniform_(self.w_dst_1.data)
       

    def forward(self, feat_in, adj, s_mask):
        adj = torch.FloatTensor(adj).cuda().unsqueeze(1)
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim
        eye=torch.eye(N,N).cuda()
        eye=eye.unsqueeze(0).unsqueeze(0)
        eye=eye.expand(batch,-1,-1,-1)
        s_mask=s_mask.float().unsqueeze(1) #(B,1,N,N)
        inter_clause=torch.where(s_mask==1,adj,s_mask)#
        outer_clause=torch.where(s_mask==0,adj,1-s_mask)#
        intra_clause=torch.where(eye==1,adj,eye)#
        
        mask2=1-inter_clause
        mask3=1-outer_clause

        feat_in_ = feat_in.unsqueeze(1) #(B,1,N,D) 
        h1 = torch.matmul(feat_in_, self.W_1) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h2 = torch.matmul(feat_in_, self.W_2) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h3 = torch.matmul(feat_in_, self.W_3) #(B,1,N,D)X(4,D,192)=(B,4,N,192)

        attn_src_1 = torch.matmul(F.tanh(h1), self.w_src_1) #(B,4,N,1)
        attn_dst_1 = torch.matmul(F.tanh(h1), self.w_dst_1) #(B,4,N,1)
        attn1 = attn_src_1.expand(-1, -1, -1, N) + attn_dst_1.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn1 = F.leaky_relu(attn1, self.leaky_alpha, inplace=True)
        attn2=torch.clone(attn1)
        attn3=torch.clone(attn1)
        
        attn2.data.masked_fill_(mask2.byte(), -99999) 
        attn3.data.masked_fill_(mask3.byte(), -99999)
        attn2 = F.softmax(attn2, dim=-1)
        attn3 = F.softmax(attn3, dim=-1)
        attn2=torch.where(inter_clause>0.0,attn2,inter_clause)
        attn3=torch.where(outer_clause>0.0,attn3,outer_clause)
        
        feat_out = torch.matmul(intra_clause,h1)+ torch.matmul(attn2, h2)+torch.matmul(attn3, h3)+ self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out,eye+attn2+attn3

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    
class GATencoderlayer(nn.Module):
    
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GATencoderlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)

        attn_src = torch.matmul(F.tanh(h), self.w_src)
        attn_dst = torch.matmul(F.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        adj = torch.FloatTensor(adj).cuda()
        mask = 1 - adj.unsqueeze(1)
        # print(attn.size())
        # print(mask.size())
        attn.data.masked_fill_(mask.byte(), -999)

        attn = F.softmax(attn, dim=-1)
        I=torch.eye(N).repeat(batch,self.att_head,1,1).cuda()#
        attn=attn+I
        feat_out = torch.matmul(attn, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out,attn

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    
class GATdecoderlayer(nn.Module):
    
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GATdecoderlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, IB,adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim
        I=torch.eye(N).repeat(batch,self.att_head,1,1).cuda()#
        b_inv,_ = torch.solve(I,IB)
        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)

        feat_out = torch.matmul(b_inv, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    
class EnhancedAttentionLayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(EnhancedAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W_1 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_2 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_3 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.b = nn.Parameter(torch.Tensor(self.out_dim)) #(192)

        self.w_src_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        self.w_dst_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_src_2 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_dst_2 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_src_3 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_dst_3 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)
        
        

    def init_gnn_param(self):
        init.xavier_uniform_(self.W_1.data)
        init.xavier_uniform_(self.W_2.data)
        init.xavier_uniform_(self.W_3.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src_1.data)
        init.xavier_uniform_(self.w_dst_1.data)
        # init.xavier_uniform_(self.w_src_2.data)
        # init.xavier_uniform_(self.w_dst_2.data)
        # init.xavier_uniform_(self.w_src_3.data)
        # init.xavier_uniform_(self.w_dst_3.data)

    def forward(self, feat_in, adj, relation, s_mask):
        adj = torch.FloatTensor(adj).cuda().unsqueeze(1)
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim
        eye=torch.eye(N,N).cuda()
        eye=eye.unsqueeze(0).unsqueeze(0)
        eye=eye.expand(batch,-1,-1,-1)
        s_mask=s_mask.float().unsqueeze(1) #(B,1,N,N)
        inter_clause=torch.where(s_mask==1,s_mask-eye,s_mask)#
        outer_clause=torch.where(s_mask==0,adj,1-s_mask)#
        intra_clause=torch.where(eye==1,adj,eye)#
        
        mask2=1-inter_clause
        mask3=1-outer_clause

        feat_in_ = feat_in.unsqueeze(1) #(B,1,N,D) 
        h1 = torch.matmul(feat_in_, self.W_1) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h2 = torch.matmul(feat_in_, self.W_2) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h3 = torch.matmul(feat_in_, self.W_3) #(B,1,N,D)X(4,D,192)=(B,4,N,192)

        attn_src_1 = torch.matmul(F.tanh(h1), self.w_src_1) #(B,4,N,1)
        attn_dst_1 = torch.matmul(F.tanh(h1), self.w_dst_1) #(B,4,N,1)
        attn1 = attn_src_1.expand(-1, -1, -1, N) + attn_dst_1.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn1 = F.leaky_relu(attn1, self.leaky_alpha, inplace=True)
        attn2=torch.clone(attn1)
        attn3=torch.clone(attn1)
        # attn_src_2 = torch.matmul(F.tanh(h2), self.w_src_2) #(B,4,N,1)
        # attn_dst_2 = torch.matmul(F.tanh(h2), self.w_dst_2) #(B,4,N,1)
        # attn2 = attn_src_2.expand(-1, -1, -1, N) + attn_dst_2.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        # attn2 = F.leaky_relu(attn2, self.leaky_alpha, inplace=True)
        
        # attn_src_3 = torch.matmul(F.tanh(h3), self.w_src_3) #(B,4,N,1)
        # attn_dst_3 = torch.matmul(F.tanh(h3), self.w_dst_3) #(B,4,N,1)
        # attn3 = attn_src_3.expand(-1, -1, -1, N) + attn_dst_3.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        # attn3 = F.leaky_relu(attn3, self.leaky_alpha, inplace=True)
    
        
        
        # print(attn.size())
        # print(mask.size())
   
        #attn1=attn1*intra_clause
        attn2.data.masked_fill_(mask2.byte(), -99999) #
        attn3.data.masked_fill_(mask3.byte(), -99999)
        attn2 = F.softmax(attn2, dim=-1)
        attn3 = F.softmax(attn3, dim=-1)
        attn2=torch.where(inter_clause>0.0,attn2,inter_clause)
        attn3=torch.where(outer_clause>0.0,attn3,outer_clause)
        
        feat_out = (torch.matmul(intra_clause, h1) +torch.matmul(attn2, h2)+torch.matmul(attn3, h3)+ self.b)/3

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    
class ActivateAttentionLayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(ActivateAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn
        self.embedding_dim=100

        self.att_head = att_head
        self.embedding=nn.Parameter(self.get_embedding())
        self.W_1 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_2 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.W_3 = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim)) #(4,768,192) 768=D
        self.b = nn.Parameter(torch.Tensor(self.out_dim)) #(192)

        self.w_src_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        self.w_dst_1 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_src_2 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_dst_2 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_src_3 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        # self.w_dst_3 = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1)) #(4,192,1)
        self.w_src_o1 = nn.Parameter(torch.Tensor(self.att_head, self.embedding_dim, 1))
        self.w_dst_o1 = nn.Parameter(torch.Tensor(self.att_head, self.embedding_dim, 1))
        # self.w_src_o2 = nn.Parameter(torch.Tensor(self.att_head, self.embedding_dim, 1))
        # self.w_dst_o2 = nn.Parameter(torch.Tensor(self.att_head, self.embedding_dim, 1))
        # self.w_src_o3 = nn.Parameter(torch.Tensor(self.att_head, self.embedding_dim, 1))
        # self.w_dst_o3 = nn.Parameter(torch.Tensor(self.att_head, self.embedding_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)
        #init.xavier_normal_(self.rs.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W_1.data)
        init.xavier_uniform_(self.W_2.data)
        init.xavier_uniform_(self.W_3.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src_1.data)
        init.xavier_uniform_(self.w_dst_1.data)
        # init.xavier_uniform_(self.w_src_2.data)
        # init.xavier_uniform_(self.w_dst_2.data)
        # init.xavier_uniform_(self.w_src_3.data)
        # init.xavier_uniform_(self.w_dst_3.data)
        init.xavier_uniform_(self.w_src_o1.data)
        init.xavier_uniform_(self.w_dst_o1.data)
        # init.xavier_uniform_(self.w_src_o2.data)
        # init.xavier_uniform_(self.w_dst_o2.data)
        # init.xavier_uniform_(self.w_src_o3.data)
        # init.xavier_uniform_(self.w_dst_o3.data)

    def get_embedding(self):
        embedding_sort=[list(np.zeros(100))]
        embedding_sort.extend([list(np.random.normal(loc=0.0,scale=0.1,size=100))for i in range(75)])
        return torch.FloatTensor(np.array(embedding_sort))
        

    def forward(self, feat_in, adj, relation,s_mask,output1, output2):
        adj = torch.FloatTensor(adj).cuda().unsqueeze(1)
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim
        eye=torch.eye(N,N).cuda()
        eye=eye.unsqueeze(0).unsqueeze(0)
        eye=eye.expand(batch,-1,-1,-1)
        s_mask=s_mask.float().unsqueeze(1) #(B,1,N,N)
        inter_clause=torch.where(s_mask==1,s_mask-eye,s_mask)#
        outer_clause=torch.where(s_mask==0,adj,1-s_mask)#
        intra_clause=torch.where(eye==1,adj,eye)#
        
        mask2=1-inter_clause
        mask3=1-outer_clause

        feat_in_ = feat_in.unsqueeze(1) #(B,1,N,D) 
        #h = torch.matmul(feat_in_, self.W) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        mask_sort=torch.sign(torch.sum(adj.squeeze(1),dim=-1).squeeze(-1)).int()#(B,N)
        output1=(torch.sort(output1,stable=True).indices.int()+1)*mask_sort
        output2=(torch.sort(output2,stable=True).indices.int()+1)*mask_sort
        output1=F.embedding(output1,self.embedding,padding_idx=0)
        output2=F.embedding(output2,self.embedding,padding_idx=0)#(B,N,embedding_dim)
        output1_=output1.unsqueeze(1)#(B,1,N,embedding_dim)
        output2_=output2.unsqueeze(1)
        output1_1=torch.matmul(F.tanh(output1_.expand(-1,self.att_head,-1,-1)),self.w_src_o1)#(B,4,N,1)
        output2_1=torch.matmul(F.tanh(output2_.expand(-1,self.att_head,-1,-1)),self.w_dst_o1)
        # output1_2=torch.matmul(F.tanh(output1_.expand(-1,self.att_head,-1,-1)),self.w_src_o2)#(B,4,N,1)
        # output2_2=torch.matmul(F.tanh(output2_.expand(-1,self.att_head,-1,-1)),self.w_dst_o2)
        # output1_3=torch.matmul(F.tanh(output1_.expand(-1,self.att_head,-1,-1)),self.w_src_o3)#(B,4,N,1)
        # output2_3=torch.matmul(F.tanh(output2_.expand(-1,self.att_head,-1,-1)),self.w_dst_o3)
        attn_add1=output1_1+output2_1.permute(0,1,3,2)
        # attn_add2=output1_2+output2_2.permute(0,1,3,2)
        # attn_add3=output1_3+output2_3.permute(0,1,3,2)
        attn_add2=torch.clone(attn_add1)
        attn_add3=torch.clone(attn_add1)

        h1 = torch.matmul(feat_in_, self.W_1) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h2 = torch.matmul(feat_in_, self.W_2) #(B,1,N,D)X(4,D,192)=(B,4,N,192)
        h3 = torch.matmul(feat_in_, self.W_3) #(B,1,N,D)X(4,D,192)=(B,4,N,192)

        attn_src_1 = torch.matmul(F.tanh(h1), self.w_src_1) #(B,4,N,1)
        attn_dst_1 = torch.matmul(F.tanh(h1), self.w_dst_1) #(B,4,N,1)
        attn1 = attn_src_1.expand(-1, -1, -1, N) + attn_dst_1.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn1 = F.leaky_relu(attn1, self.leaky_alpha, inplace=True)
        attn2=torch.clone(attn1)
        attn3=torch.clone(attn1)
        # attn_src_2 = torch.matmul(F.tanh(h2), self.w_src_2) #(B,4,N,1)
        # attn_dst_2 = torch.matmul(F.tanh(h2), self.w_dst_2) #(B,4,N,1)
        # attn2 = attn_src_2.expand(-1, -1, -1, N) + attn_dst_2.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        # attn2 = F.leaky_relu(attn2, self.leaky_alpha, inplace=True)
        
        # attn_src_3 = torch.matmul(F.tanh(h3), self.w_src_3) #(B,4,N,1)
        # attn_dst_3 = torch.matmul(F.tanh(h3), self.w_dst_3) #(B,4,N,1)
        # attn3 = attn_src_3.expand(-1, -1, -1, N) + attn_dst_3.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        # attn3 = F.leaky_relu(attn3, self.leaky_alpha, inplace=True)
    
        
        
        # print(attn.size())
        # print(mask.size())
 
        #attn1=attn1*intra_clause+attn_add1*intra_clause
        attn2.data.masked_fill_(mask2.byte(), -99999) 
        attn_add2.data.masked_fill_(mask2.byte(), -99999)
        attn3.data.masked_fill_(mask3.byte(), -99999)
        attn_add3.data.masked_fill_(mask3.byte(), -99999)
        attn2_all = F.softmax(attn2+attn_add2, dim=-1)
        attn3_all = F.softmax(attn3+attn_add3, dim=-1)
        attn2_all=torch.where(inter_clause>0.0,attn2_all,inter_clause)
        attn3_all=torch.where(outer_clause>0.0,attn3_all,outer_clause)
        
        feat_out = (torch.matmul(intra_clause, h1) +torch.matmul(attn2_all, h2)+torch.matmul(attn3_all, h3)+ self.b)/3

        
        #attn_add=self.rs(attn_add.unsqueeze(-1)).squeeze(-1)


        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    

