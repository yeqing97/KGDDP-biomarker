import torch.nn as nn
import dgl.nn as dglnn
import torch


class KGDDP(nn.Module):
    def __init__(self, in_feats, hid_feats, aggregator_type,feat_drop,device,pid_fea_sc, drug_fea, pro_entity_df, pathway_entity_df, go_entity_df):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            'ppi':dglnn.SAGEConv(in_feats, in_feats,aggregator_type=aggregator_type,feat_drop=feat_drop),
            'pro_path':dglnn.SAGEConv(in_feats, in_feats,aggregator_type=aggregator_type,feat_drop=feat_drop),
            'pro_go':dglnn.SAGEConv(in_feats, in_feats,aggregator_type=aggregator_type,feat_drop=feat_drop),
            'dpi':dglnn.SAGEConv(in_feats, in_feats,aggregator_type=aggregator_type,feat_drop=feat_drop),
            },aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            'pro_pid':dglnn.GraphConv(in_feats,in_feats,norm='both', weight=True, bias=True, allow_zero_in_degree=True),
            },aggregate='sum')
        self.pid_embd = torch.from_numpy(pid_fea_sc).type(torch.float32).to(device)  
        self.drug_embd = torch.from_numpy(drug_fea).type(torch.float32).to(device)
        self.pro_embd = nn.Embedding(len(pro_entity_df),in_feats)
        self.path_embd = nn.Embedding(len(pathway_entity_df),in_feats)
        self.go_embd = nn.Embedding(len(go_entity_df),in_feats)
        self.W_in_drug = nn.Linear(1024, in_feats)
        self.W_in_pid = nn.Linear(pid_fea_sc.shape[1], in_feats)
        self.mlp_decoder = nn.Sequential(
                    nn.Linear(in_feats*2, hid_feats),
                    nn.ReLU(),
                    nn.Linear(hid_feats, 1),
                    )
        self.classifier_d = nn.Sequential(
                    nn.Linear(in_feats,hid_feats),
                    nn.ReLU(),
                    nn.Linear(hid_feats,3))
        self.mlp_decoder_dpi = nn.Sequential(
                    nn.Linear(in_feats*2, hid_feats),
                    nn.ReLU(),
                    nn.Linear(hid_feats, 1),
                    )
        self.mlp_decoder_propath = nn.Sequential(
                    nn.Linear(in_feats*2, hid_feats),
                    nn.ReLU(),
                    nn.Linear(hid_feats, 1),
                    )

    def forward(self, g, h_head_idx, h_tail1_id, h_tail2_id,t_tail_idx, t_head1_id, t_head2_id):
        node_feats = {}
        pid_in_feats = self.W_in_pid(self.pid_embd)
        dr_in_feats = self.W_in_drug(self.drug_embd)

        node_feats['protein'] = self.pro_embd.weight
        node_feats['pathway'] = self.path_embd.weight
        node_feats['go'] = self.go_embd.weight
        node_feats['drug'] = dr_in_feats
        node_feats = self.conv1(g, node_feats)

        node_feats['pid'] = pid_in_feats
        node_feats_ = self.conv2(g, node_feats,node_feats)
        #print(node_feats_)

        node_feats['pid'] = node_feats_['pid'] 
        h_head_feats = node_feats['protein'][h_head_idx]
        h_tail1_feats = node_feats['pid'][h_tail1_id]
        h_tail2_feats = node_feats['pid'][h_tail2_id]
        h_score_pc1 = self.mlp_decoder(torch.cat([h_head_feats, h_tail1_feats], 1))
        h_score_pc2 = self.mlp_decoder(torch.cat([h_head_feats, h_tail2_feats], 1))

        t_tail_feats = node_feats['pid'][t_tail_idx]
        h_head1_feats = node_feats['protein'][t_head1_id]
        h_head2_feats = node_feats['protein'][t_head2_id]
        t_score_pc1 = self.mlp_decoder(torch.cat([h_head1_feats, t_tail_feats], 1))
        t_score_pc2 = self.mlp_decoder(torch.cat([h_head2_feats, t_tail_feats], 1))

        node_feats['drug'] = dr_in_feats 

        return (h_score_pc1 - h_score_pc2),(t_score_pc1 - t_score_pc2), node_feats




