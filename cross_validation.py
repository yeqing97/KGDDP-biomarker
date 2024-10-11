import pandas as pd
import numpy as np
from tqdm import tqdm
from data_utils import preprocess_data,construct_graph
from model import KGDDP
from train_model import train_one_epoch, evaluate_model
import torch
from sklearn.model_selection import train_test_split,StratifiedKFold
import optuna

# Set the path to the data directory
path = './data/'

# Load and preprocess data...
kg = pd.read_csv(path+'kg_del_selfloop.csv')
pro_path_neg_sp = pd.read_csv(path + 'human_neg_pathpro.csv')
dpi_neg = pd.read_csv(path + 'neg_dpi_df_t10.csv')
fp_df = pd.read_csv(path + 'bdki_db_gdsc_fp.csv')

exp_triples = pd.read_csv(path + 'gse169568/gse169568_exp_triples_1128.csv')
exp_triples_graph = pd.read_csv(path + 'gse169568/gse169568_graph_pair.csv')
exp_input = pd.read_csv(path + 'gse169568/gse169568_exp_1444.csv')
exp_input.columns = ['pid']+list(exp_input.columns[1:])

dls = pd.read_csv(path + 'gse169568/group_info.csv')
dls.columns = ['pid']+list(dls.columns[1:])
dls = pd.merge(dls, exp_input['pid'])

# Preprocess data
processed_data = preprocess_data(kg, exp_triples, exp_triples_graph, exp_input,
                                 pro_path_neg_sp,dpi_neg,fp_df,dls)
kg,dls,pid_entity_df,drug_entity_df,pro_entity_df,go_entity_df,pathway_entity_df,\
drug_fea,dpi_pos,pro_path_pos,pid_fea_sc = processed_data

# Construct graph
graph = construct_graph(kg,exp_triples_graph,
                        pid_entity_df,drug_entity_df,pro_entity_df,go_entity_df,pathway_entity_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Training loop
def train_loop(params,train_pid, val_pid, train_pid_label, val_pid_label,iter):
    exp_triples_train = pd.merge(exp_triples,pd.DataFrame(train_pid,columns=['tail_id']))
    exp_triples_test = pd.merge(exp_triples,pd.DataFrame(val_pid,columns=['tail_id']))

    in_feats = params['in_feats']
    hid_feats = params['hid_feats']
    aggregator_type = params['aggregator_type']
    feat_drop = params['feat_drop']
    patience = params['patience']

    # Instantiate the model
    model = KGDDP(in_feats, hid_feats, aggregator_type,feat_drop,
                device,pid_fea_sc, drug_fea, pro_entity_df, pathway_entity_df, go_entity_df).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=10e-6)

    loss_thre = 0.4
    stop_metrics = 0
    early_stop_epoch = 0
    val_spearman = 0

    for epoch in tqdm(range(1000)):
        model,node_feats,loss_sum,d_score,exp_contrast_test_head,exp_contrast_test_tail = train_one_epoch(model, opt, graph,
                    exp_triples_train, exp_triples_test,
                    pro_path_pos,pro_path_neg_sp,dpi_pos,dpi_neg,
                    train_pid,train_pid_label,
                    params,epoch,device,loss_thre,
                    val_spearman)
        loss_sum_mean, val_corr, val_spearman, train_d_acc, val_d_roc_auc, val_d_acc = evaluate_model(model,node_feats,exp_contrast_test_head,exp_contrast_test_tail,epoch,
                    loss_sum,d_score,train_pid_label,val_pid,val_pid_label)
        
        ### Early stopping
        if val_spearman > loss_thre:
            cur_metric = val_spearman + val_d_acc
        else:
            cur_metric = val_spearman
        if epoch > 20:
            if epoch % 2 == 0:
                if cur_metric >= stop_metrics:
                    stop_metrics = cur_metric
                    early_stop_epoch=0
                    print(cur_metric)
                    # Save model and metrics
                    np.save('./gse66407_cv_'+str(iter)+'.npy', node_feats)
                    torch.save(model.cpu(), './gse66407_cv_'+str(iter)+'.pth')
                else:
                    early_stop_epoch = early_stop_epoch + 1
                    print('early_stop_epoch: '+str(early_stop_epoch))
                    if early_stop_epoch==patience:
                        print('---------------------Finish------------------------')
                        break

    return val_d_acc


# Train-validation-test split
params = {'in_feats': 344, 'hid_feats': 168, 
          'aggregator_type': 'mean', 'feat_drop': 0.2, 
          'patience': 18, 'batch_size': 4777, 'sample_propath_size': 8044,
          'propath_mlti': 0.5, 'sample_dpi_size': 2082, 'dpi_mlti': 1,'w_path_pro': 0.1, 'd_w': 0.1,'w_dpi': 0.01}

iter = 0
skf = StratifiedKFold(n_splits=5)

val_accs = []
for train_index, val_index in skf.split(dls['tail_id'].values, dls['d_label'].values):
    iter += 1
    print('########################## New Split ########################')
    print(iter)
    print('########################## Now Start ########################')
    train_pid = dls['tail_id'].values[train_index]
    train_pid_label = dls['d_label'].values[train_index]
    val_pid = dls['tail_id'].values[val_index]
    val_pid_label = dls['d_label'].values[val_index]
    print(np.unique(val_pid_label,return_counts=True))
    val_acc = train_loop(params,train_pid, val_pid, train_pid_label, val_pid_label,iter)
    val_accs.append(val_acc)

print(np.array(val_accs).mean())
print(np.array(val_accs).std())