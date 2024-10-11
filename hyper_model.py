import pandas as pd
from tqdm import tqdm
from data_utils import preprocess_data,construct_graph
from model import KGDDP
from train_model import train_one_epoch, evaluate_model
import torch
from sklearn.model_selection import train_test_split
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

# Train-validation-test split
train_val_pid, test_pid, train_val_pid_label,test_pid_label = train_test_split(dls['tail_id'].values,dls['d_label'].values,test_size=0.2,random_state=42,stratify=dls['d_label'].values)
train_pid,     val_pid,  train_pid_label,    val_pid_label  = train_test_split(train_val_pid,    train_val_pid_label,  test_size=0.2,random_state=42,stratify=train_val_pid_label)
exp_triples_train = pd.merge(exp_triples,pd.DataFrame(train_pid,columns=['tail_id']))
exp_triples_test = pd.merge(exp_triples,pd.DataFrame(val_pid,columns=['tail_id']))

# Training loop
def train_loop(params):

    in_feats = params['in_feats']
    hid_feats = params['hid_feats']
    aggregator_type = params['aggregator_type']
    feat_drop = params['feat_drop']
    patience = params['patience']

    # Instantiate the model
    model = KGDDP(in_feats, hid_feats, aggregator_type,feat_drop,
                device,pid_fea_sc, drug_fea, pro_entity_df, pathway_entity_df, go_entity_df).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=10e-6)

    loss_thre = 0.7
    stop_metrics = 0
    best_metric = 0
    early_stop_epoch = 0
    best_val_acc = 0
    val_spearman = 0

    for epoch in tqdm(range(100)):
        model,node_feats,loss_sum,d_score,exp_contrast_test_head,exp_contrast_test_tail = train_one_epoch(model, opt, graph,
                    exp_triples_train, exp_triples_test,
                    pro_path_pos,pro_path_neg_sp,dpi_pos,dpi_neg,
                    train_pid,train_pid_label,
                    params,epoch,device,loss_thre,
                    val_spearman)
        loss_sum_mean, val_corr, val_spearman, train_d_acc, val_d_roc_auc, val_d_acc = evaluate_model(model,node_feats,
                    exp_contrast_test_head,exp_contrast_test_tail,epoch,
                    loss_sum,d_score,train_pid_label,val_pid,val_pid_label)
        
        ### Early stopping
        if val_spearman > loss_thre:
            cur_metric = val_spearman + val_d_acc
        else:
            cur_metric = val_spearman
        if epoch > 20 and cur_metric >= stop_metrics:
            best_metric = cur_metric
            best_val_acc = val_d_acc
            print(best_val_acc)
            # Save model and metrics
            # np.save('eval_emb_file.npy', node_feats)
            # torch.save(het_gnn.cpu(), 'eval_model_file.pth')
        if epoch % 2 == 0:
            stop_metrics = best_metric
            if cur_metric >= stop_metrics:
                early_stop_epoch=0
            else:
                early_stop_epoch = early_stop_epoch + 1
                print('early_stop_epoch: '+str(early_stop_epoch))
                if early_stop_epoch==patience:
                    print('---------------------Finish------------------------')
                    break

    return best_val_acc


def objective(trial):
    # Model hyperparameters
    params = {
        'in_feats': trial.suggest_int('n_feats', 256, 512), 
        'hid_feats': trial.suggest_int('hid_feats', 128, 256), 
        'aggregator_type': trial.suggest_categorical('aggregator_type', ['gcn', 'mean']), 
        'feat_drop': trial.suggest_categorical('feat_drop', [0, 0.1, 0.2, 0.3]), 
        'patience':trial.suggest_int('patience', 5,20),
        'batch_size':trial.suggest_int('batch_size', 1000, 10000),
        'sample_propath_size':trial.suggest_int('sample_propath_size', 1000, 10000),
        'propath_mlti':trial.suggest_categorical('propath_mlti', [0.1, 0.5, 1, 2]),
        'sample_dpi_size':trial.suggest_int('sample_dpi_size', 1000, 10000), 
        'dpi_mlti':trial.suggest_categorical('dpi_mlti', [0.1, 0.5, 1, 2]),
        'w_path_pro':trial.suggest_categorical('w_path_pro', [0.01, 0.1, 1]),
        'd_w':trial.suggest_categorical('d_w', [0.01, 0.1, 1]),
        'w_dpi':trial.suggest_categorical('w_dpi', [0.01, 0.1, 1])
          }

    best_val_acc = train_loop(params)

    return best_val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50,show_progress_bar=True,gc_after_trial=True,n_jobs=1)