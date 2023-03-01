import torch
import torch.nn as nn

from dgl.nn.pytorch import EGATConv
import dgl.function as fn
class MLP(nn.Module):
    def __init__(self, in_feats, class_num=2, device='cpu'):
        super(MLP, self).__init__()
        self.device = device
        self.cuda = True if device != 'cpu' else False
        self.Sequential = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.5),
            nn.Linear(64, class_num),
        )

    def forward(self, feature):
        out = self.Sequential(feature)
        return out

class EGAT_encoder(nn.Module):
    def __init__(self, node_feats, edge_feats, encode_feats,  heads, hidden_ly=2, device = 'cpu'):
        super(EGAT_encoder, self).__init__()
        self.device = device
        self.cuda = True if device != "cpu" else False
        in_moudle = nn.ModuleList([EGATConv(in_node_feats=node_feats,
                                            in_edge_feats=edge_feats,
                                            out_node_feats=encode_feats,
                                            out_edge_feats=encode_feats,
                                            num_heads=heads).to(device)])
        hidden_module = nn.ModuleList([EGATConv(in_node_feats=encode_feats,
                                                in_edge_feats=encode_feats,
                                                out_node_feats=encode_feats,
                                                out_edge_feats=encode_feats,
                                                num_heads=heads).to(device) for _ in range(hidden_ly)])
        self.layers = in_moudle + hidden_module
        # self.mlp = MLP(encode_feats)
        self.mlp2 = MLP(encode_feats*2)

    def forward(self, g, node_feat, edge_feat):
        for layer in self.layers:
            node_feat, edge_feat = layer(g, node_feat, edge_feat)
            node_feat, edge_feat = node_feat.mean(dim=1), edge_feat.mean(dim=1)
        # g.ndata['feat'] = node_feat
        # g.edata['feat'] = edge_feat
        # return self.mlp(g.ndata['feat'])

        with g.local_scope():
            g.ndata['h'] = node_feat
            g.edata['w'] = edge_feat
            g.update_all(
                message_func = fn.copy_e('w', 'm'),
                reduce_func = fn.mean('m', 'e_N'),
            )
            e_N = g.ndata['e_N']
            h_total = torch.cat([g.ndata['h'], e_N], dim = 1)
            return self.mlp2(h_total)


        

class NodeGNN(nn.Module):
    def __init__(self, node_feats, edge_feats, encode_feats, heads, hidden_ly, device):
        super(NodeGNN, self).__init__()
        self.encoder = EGAT_encoder(node_feats,edge_feats,encode_feats,heads,hidden_ly,device)
        self.device = device

    def forward(self, g, node_feat, edge_feat):
        h = self.encoder(g, node_feat, edge_feat)
        return h
