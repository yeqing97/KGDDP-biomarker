import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_auc_score,accuracy_score

import warnings
warnings.filterwarnings('ignore')

criterion = nn.CrossEntropyLoss()
max_grad_norm = 1.0  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])[:,0]
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels.to(device))


def get_constrast_df(data,merge_id):
    merged_df = pd.merge(data, data, on=[merge_id], suffixes=['_1', '_2'])
    merged_df['contrast_value'] = merged_df['exp_1'] - merged_df['exp_2']
    thre = 0
    pos_m = merged_df[merged_df['contrast_value']>thre]
    pos_m['constrast_label'] = 1
    neg_m = merged_df[merged_df['contrast_value']<-thre]
    neg_m['constrast_label'] = 0
    constrast_df = pd.concat([pos_m,neg_m])
    constrast_df.index = range(len(constrast_df))
    return constrast_df

def func_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def train_one_epoch(model, opt, g,
                exp_triples_train, exp_triples_test,
                pro_path_pos,pro_path_neg_sp,dpi_pos,dpi_neg,
                train_pid,train_pid_label,
                params,seed,device,loss_thre,
                val_spearman):
    # Training loop...
    model.train()
    loss_sum = []
    batch_size = params['batch_size']
    sample_propath_size = params['sample_propath_size']
    propath_mlti = params['propath_mlti']
    sample_dpi_size = params['sample_dpi_size']
    dpi_mlti = params['dpi_mlti']
    w_path_pro = params['w_path_pro']
    d_w = params['d_w']
    w_dpi = params['w_dpi']

    mrna_triples_train = exp_triples_train.sample(n=batch_size, replace=False, random_state=seed)
    mrna_triples_train.index = range(len(mrna_triples_train))
    exp_contrast_train_tail = get_constrast_df(mrna_triples_train,'tail_id').sample(n=batch_size, replace=True, random_state=0)
    exp_contrast_train_head = get_constrast_df(mrna_triples_train,'head_id').sample(n=batch_size, replace=True, random_state=0)
    mrna_triples_test = exp_triples_test.sample(n=batch_size, replace=False, random_state=seed)        
    mrna_triples_test.index = range(len(mrna_triples_test))
    exp_contrast_test_head = get_constrast_df(mrna_triples_test,'head_id').sample(n=batch_size, replace=True, random_state=0)
    exp_contrast_test_tail = get_constrast_df(mrna_triples_test,'tail_id').sample(n=batch_size, replace=True, random_state=0)

    pro_path_pos_sample = pro_path_pos.sample(n=sample_propath_size, replace=False, random_state=seed)
    pro_path_neg_sp_sample = pro_path_neg_sp.sample(n=int(sample_propath_size*propath_mlti), replace=False, random_state=seed)
    dpi_pos_sample = dpi_pos.sample(n=sample_dpi_size, replace=False, random_state=seed)
    dpi_neg_sample = dpi_neg.sample(n=int(sample_dpi_size*dpi_mlti), replace=False, random_state=seed)

    h_head_idx, h_tail1_id, h_tail2_id, h_label = exp_contrast_train_head['head_id'].values,exp_contrast_train_head['tail_id_1'].values, \
                                                    exp_contrast_train_head['tail_id_2'].values,exp_contrast_train_head['constrast_label'].values
    t_tail_idx, t_head1_id, t_head2_id, t_label = exp_contrast_train_tail['tail_id'].values,exp_contrast_train_tail['head_id_1'].values, \
                                                    exp_contrast_train_tail['head_id_2'].values,exp_contrast_train_tail['constrast_label'].values
    h_pair, t_pair,node_feats = model(g, h_head_idx, h_tail1_id, h_tail2_id,t_tail_idx, t_head1_id, t_head2_id)


    dpi_pos_pred = model.mlp_decoder_dpi(torch.cat([node_feats['drug'][dpi_pos_sample['head_id'].values], 
                                        node_feats['protein'][dpi_pos_sample['tail_id'].values]], 1))
    dpi_neg_pred = model.mlp_decoder_dpi(torch.cat([node_feats['drug'][dpi_neg_sample['head_id'].values], 
                                        node_feats['protein'][dpi_neg_sample['tail_id'].values]], 1))   
    pro_path_pos_pred = model.mlp_decoder_propath(torch.cat([node_feats['protein'][pro_path_pos_sample['tail_id'].values], 
                                        node_feats['pathway'][pro_path_pos_sample['head_id'].values]], 1))
    pro_path_neg_pred = model.mlp_decoder_propath(torch.cat([node_feats['protein'][pro_path_neg_sp_sample['tail_id'].values], 
                                        node_feats['pathway'][pro_path_neg_sp_sample['head_id'].values]], 1))
    
    loss_dpi = compute_loss(dpi_pos_pred, dpi_neg_pred)
    loss_pro_path = compute_loss(pro_path_pos_pred, pro_path_neg_pred)
    loss_pair_h = F.binary_cross_entropy_with_logits(h_pair[:,0], torch.tensor(h_label).float().to(device))
    loss_pair_t = F.binary_cross_entropy_with_logits(t_pair[:,0], torch.tensor(t_label).float().to(device))
    d_score = model.classifier_d(node_feats['pid'][train_pid])
    d_loss = criterion(d_score,torch.tensor(train_pid_label,dtype=torch.int64).to(device)) 
    if val_spearman > loss_thre:
        loss = loss_pair_h + loss_pair_t + w_path_pro * loss_pro_path + d_w * d_loss + w_dpi * loss_dpi 
    else:
        loss = loss_pair_h + loss_pair_t + w_path_pro * loss_pro_path + w_dpi * loss_dpi
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    opt.step()
    loss_sum.append(loss.item())
    return model,node_feats,loss_sum,d_score,exp_contrast_test_head,exp_contrast_test_tail

def evaluate_model(model,node_feats,exp_contrast_test_head,exp_contrast_test_tail,epoch,
                   loss_sum,d_score,train_pid_label,val_pid,val_pid_label):
    # Evaluation...
    model.eval()
    loss_sum_mean = np.array(loss_sum).sum()

    #val_pro_id, val_pid_idx, val_y = exp_contrast_test['head_id'].values,exp_contrast_test['tail_id'].values, exp_contrast_test['exp'].values
    val_h_head_idx, val_h_tail1_id, val_h_tail2_id, val_h_label = exp_contrast_test_head['head_id'].values,exp_contrast_test_head['tail_id_1'].values, \
                                                    exp_contrast_test_head['tail_id_2'].values,exp_contrast_test_head['constrast_label'].values
    val_t_tail_idx, val_t_head1_id, val_t_head2_id, val_t_label = exp_contrast_test_tail['tail_id'].values,exp_contrast_test_tail['head_id_1'].values, \
                                                    exp_contrast_test_tail['head_id_2'].values,exp_contrast_test_tail['constrast_label'].values
    h_pc1_val_preds  = model.mlp_decoder(torch.cat([node_feats['protein'][val_h_head_idx],node_feats['pid'][val_h_tail1_id]],1)).detach().cpu().numpy()
    h_pc2_val_preds  = model.mlp_decoder(torch.cat([node_feats['protein'][val_h_head_idx],node_feats['pid'][val_h_tail2_id]],1)).detach().cpu().numpy()
    h_pc_val_preds = h_pc1_val_preds - h_pc2_val_preds
    t_pc1_val_preds  = model.mlp_decoder(torch.cat([node_feats['protein'][val_t_head1_id],node_feats['pid'][val_t_tail_idx]],1)).detach().cpu().numpy()
    t_pc2_val_preds  = model.mlp_decoder(torch.cat([node_feats['protein'][val_t_head2_id],node_feats['pid'][val_t_tail_idx]],1)).detach().cpu().numpy()
    t_pc_val_preds = t_pc1_val_preds - t_pc2_val_preds
    pc_val_preds = np.concatenate([np.array(h_pc_val_preds).reshape(1,-1)[0],np.array(t_pc_val_preds).reshape(1,-1)[0]])
    pc_val_label = np.concatenate([val_h_label,val_t_label])
    #pc_val_preds = np.array(t_pc_val_preds).reshape(1,-1)[0]
    #pc_val_label = val_t_label
    d_val_pred = model.classifier_d(node_feats['pid'][val_pid]).detach().cpu().numpy()
    d_val_predlabel = np.argmax(d_val_pred, axis=1)

    val_corr, _p = pearsonr(pc_val_preds,pc_val_label)
    val_spearman, _p = spearmanr(pc_val_preds,pc_val_label)

    d_train_predlabel = np.argmax(d_score.detach().cpu().numpy(), axis=1)
    train_d_acc = accuracy_score(train_pid_label,d_train_predlabel)
    val_d_roc_auc = roc_auc_score(val_pid_label,func_softmax(d_val_pred),multi_class='ovo')
    val_d_acc = accuracy_score(val_pid_label,d_val_predlabel,)

    print("epoch: {:04d} | loss: {:.4f} | val_corr: {:.4f} | val_spearman: {:.4f} | train_d_acc: {:.4f} | val_d_roc_auc: {:.4f} | val_d_acc: {:.4f} ".format(
        epoch, loss_sum_mean, val_corr, val_spearman, train_d_acc, val_d_roc_auc, val_d_acc))
    return loss_sum_mean, val_corr, val_spearman, train_d_acc, val_d_roc_auc, val_d_acc