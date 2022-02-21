import torch.nn as nn
import torch.nn.functional as F
from predictors.layers import GraphConvolution
from torch.nn import init
import torch

# nfeat=len(__dataset[0]['operations'][0]) + 1   # nfeat=num_features: anzahl spaltenelemente von operations + 1
# ifsigmoid=ifsigmoid
# layer_size = 8

# x = torch.rand(1,3)

# embeddings_cnn = torch.rand(1,64)
# embeddings_rhn = torch.rand(1,64)
# embeddings = torch.cat((embeddings_cnn, embeddings_rhn), dim=0)
# embeddings 
# embeddings = torch.flatten(embeddings, start_dim= 1) 

# fc = torch.nn.Linear(128, 1)

# x = fc(embeddings) # hat shape [1, 1]



class GCN(nn.Module):
    def __init__(self, nfeat_cnn, nfeat_rhn, ifsigmoid, layer_size = 64):
        super(GCN, self).__init__()
        self.ifsigmoid = ifsigmoid
        self.size = layer_size
        
        # modules of cnn gcn
        self.gc1_cnn = GraphConvolution(nfeat_cnn, self.size)
        self.gc2_cnn = GraphConvolution(self.size, self.size)
        self.gc3_cnn = GraphConvolution(self.size, self.size)
        self.gc4_cnn = GraphConvolution(self.size, self.size)
        self.bn1_cnn = nn.BatchNorm1d(self.size)
        self.bn2_cnn = nn.BatchNorm1d(self.size)
        self.bn3_cnn = nn.BatchNorm1d(self.size)
        self.bn4_cnn = nn.BatchNorm1d(self.size)
        
        # modules of cnn gcn
        self.gc1_rhn = GraphConvolution(nfeat_rhn, self.size)
        self.gc2_rhn = GraphConvolution(self.size, self.size)
        self.gc3_rhn = GraphConvolution(self.size, self.size)
        self.gc4_rhn = GraphConvolution(self.size, self.size)
        self.bn1_rhn = nn.BatchNorm1d(self.size)
        self.bn2_rhn = nn.BatchNorm1d(self.size)
        self.bn3_rhn = nn.BatchNorm1d(self.size)
        self.bn4_rhn = nn.BatchNorm1d(self.size)
        
        
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.size*2, 1)
        self.init_weights()

    def init_weights(self):
        
        # initialize weights of cnn gcn part
        init.uniform_(self.gc1_cnn.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2_cnn.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3_cnn.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4_cnn.weight, a=-0.05, b=0.05)
        
        # initialize weights of rhn gcn part
        init.uniform_(self.gc1_rhn.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2_rhn.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3_rhn.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4_rhn.weight, a=-0.05, b=0.05)


    def forward(self, feat_cnn, adj_cnn, feat_rhn, adj_rhn, extract_embedding=False):
       
        x = F.relu(self.bn1_cnn(self.gc1_cnn(feat_cnn, adj_cnn).transpose(2, 1))) # feat_cnn have shape [3,12,7] 
        x = x.transpose(1, 2)
        x = F.relu(self.bn2_cnn(self.gc2_cnn(x, adj_cnn).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn3_cnn(self.gc3_cnn(x, adj_cnn).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn4_cnn(self.gc4_cnn(x, adj_cnn).transpose(2, 1)))
        x = x.transpose(1, 2)
        embeddings_cnn = x[:, x.size()[1] - 1, :] # [4,64] if we have 4 batches
        
        x = F.relu(self.bn1_rhn(self.gc1_rhn(feat_rhn, adj_rhn).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn2_rhn(self.gc2_rhn(x, adj_rhn).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn3_rhn(self.gc3_rhn(x, adj_rhn).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn4_rhn(self.gc4_rhn(x, adj_rhn).transpose(2, 1)))
        x = x.transpose(1, 2)
        embeddings_rhn = x[:, x.size()[1] - 1, :] # [4,64]
        
        # embeddings_cnn = torch.rand(4,64)
        # embeddings_rhn = torch.rand(4,64)

        
        # concatenate both embeddings
        embeddings = torch.cat((embeddings_cnn, embeddings_rhn), dim=1)

        x = self.fc(embeddings)
        if extract_embedding:
            return embeddings
        if self.ifsigmoid:
            return self.sigmoid(x)
        else:
            return x


class MLP(nn.Module):
    def __init__(self, nfeat, ifsigmoid, layer_size = 64):
        super(MLP, self).__init__()
        self.size = layer_size
        self.ifsigmoid = ifsigmoid
        self.fc1 = nn.Linear(nfeat, self.size)
        self.fc2 = nn.Linear(self.size, self.size)
        self.fc3 = nn.Linear(self.size, self.size)
        self.fc4 = nn.Linear(self.size, self.size)
        self.fc5 = nn.Linear(self.size, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        init.uniform_(self.fc1.weight, a=-0.05, b=0.05)
        self.fc1.bias.data.fill_(0)
        init.uniform_(self.fc2.weight, a=-0.05, b=0.05)
        self.fc2.bias.data.fill_(0)
        init.uniform_(self.fc3.weight, a=-0.05, b=0.05)
        self.fc3.bias.data.fill_(0)
        init.xavier_uniform(self.fc4.weight, gain=nn.init.calculate_gain('relu'))
        self.fc4.bias.data.fill_(0)
        self.fc5.bias.data.fill_(0)

    def forward(self, x, extract_embedding=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        embedding = self.fc4(x)
        x = self.fc5(embedding)
        if extract_embedding:
            return embedding
        if self.ifsigmoid:
            return self.sigmoid(x)
        else:
            return x


class LSTM(nn.Module):
    def __init__(self, nfeat, timestep):
        self.emb_dim = 100
        self.hidden_dim = 100
        self.timestep = timestep
        super(LSTM, self).__init__()
        self.adj_emb = nn.Embedding(2, embedding_dim=self.emb_dim)
        init.uniform_(self.adj_emb.weight, a=-0.1, b=0.1)
        self.op_emb = nn.Embedding(nfeat, embedding_dim=self.emb_dim)
        init.uniform_(self.op_emb.weight, a=-0.1, b=0.1)
        self.rnn = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim * self.timestep, 1)

    def forward(self, x, adj):
        op = x
        adj_embed = self.adj_emb(adj)
        op_embed = self.op_emb(op)
        embed = torch.cat((adj_embed, op_embed), 1)
        out, (h_n, c_n) = self.rnn(embed)
        out = out.contiguous().view(-1, out.shape[1] * out.shape[2])
        out = self.fc(out)
        return out


if __name__ == "__main__":
    lstm = LSTM(8, 56)
    lstm = lstm.cuda()
    adj = [[1] * 49] * 128
    adj = torch.tensor(adj)
    op = torch.tensor([[1, 2, 3, 4, 5, 6, 7]] * 128)
    adj, op = adj.cuda(), op.cuda()
    out = lstm(op, adj)
