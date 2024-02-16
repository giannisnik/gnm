import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):    
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.fc1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for i in range(self.n_layers):
            nn.init.kaiming_normal_(self.fc1[i].weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
    
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.fc1[i](x))
            x = self.dropout(x)

        out = self.fc2(x)
        if self.output_dim == 1:
            out = out.squeeze()
        return out


class GNM(nn.Module):
    def __init__(self, n_nodes, n_layers, idx_input, idx_output, trainable_features, dropout, device):    
        super(GNM, self).__init__()
        self.n_layers = n_layers
        self.idx_input = idx_input
        self.idx_output = idx_output
        self.n_nodes = n_nodes
        if trainable_features:
            self.embeddings = nn.Parameter(torch.randn(1, n_nodes-len(idx_input)))
        else:
            self.embeddings = torch.zeros(1, n_nodes-len(idx_input), requires_grad=False, device=device)
        self.adj = nn.Parameter(torch.FloatTensor(self.n_layers, self.n_nodes, self.n_nodes))
        self.bias = nn.Parameter(torch.FloatTensor(self.n_layers, self.n_nodes))
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.adj)
        nn.init.kaiming_normal_(self.bias)

    def forward(self, x):
        t = self.embeddings.repeat(x.size(0), 1)
        x = torch.cat((x,t), dim=1)
        for i in range(self.n_layers-1):
            x = torch.mm(x, self.adj[i,:,:]) + self.bias[i,:]
            x = self.relu(x)
            x = self.dropout(x)
            
        x = torch.mm(x, self.adj[-1,:,:]) + self.bias[-1,:]

        out = x[:,self.idx_output]
        return out.squeeze()
