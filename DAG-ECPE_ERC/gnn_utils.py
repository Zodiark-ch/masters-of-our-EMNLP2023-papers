from tkinter import E
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
        self.efc=nn.Linear(1,1)
        self.sfc=nn.Linear(1,1)

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
        attn_weight_e=torch.clone(attn_weight).view(-1,1)
        attn_weight_s=torch.clone(attn_weight).view(-1,1)
        e=self.efc(attn_weight_e).view(attn_weight.size()[0],1,-1)
        s=self.sfc(attn_weight_s).view(attn_weight.size()[0],1,-1)
        V0 = self.Wr0(V) # (B, N, D)
        V1 = self.Wr1(V) # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()   # (B, N, 1)
        V = V0 * s_mask + V1 * (1 - s_mask)#V=(B,N,D)
        #PA=torch.bmm(attn_weight, V)
        attn_sum = Q[:,0,:].unsqueeze(1)-torch.bmm(attn_weight, V) 
        attn_sum=attn_sum.squeeze(1)#(B,D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum,e,s



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

        attn_sum = Q.unsqueeze(1)+torch.bmm(adj, V) 
        attn_sum=attn_sum.squeeze(1)#(B,D)
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