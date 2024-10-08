import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dgl


def preprocess_data(kg, exp_triples, exp_triples_graph, exp_input,pro_path_neg_sp,dpi_neg,fp_df,dls):
    # Further preprocessing...
    pid_entity = exp_input['pid'].unique()
    pid_entity_df = pd.DataFrame(pid_entity,columns=['entity'])
    pid_entity_df['code'] = range(len(pid_entity_df))

    dls['tail_id'] = pd.merge(dls,pid_entity_df,how='right',left_on='pid',right_on='entity')['code'].values
    dls = dls.dropna(subset=['tail_id'])
    dls.index= range(len(dls))
    class_to_label = {cls: label for label, cls in enumerate(dls['Diagnosis'].unique())}
    print(class_to_label)
    dls['d_label'] = dls['Diagnosis'].map(class_to_label)
    ### drug_entity_process
    drug_entity = pd.concat([kg[kg['relation']=='DPI']['head'],]).unique()
    drug_entity_df = pd.DataFrame(drug_entity,columns=['entity'])
    drug_entity_df['code'] = range(len(drug_entity_df))
    drug_fea = pd.merge(drug_entity_df,fp_df,how='left',on='entity').fillna(0).iloc[:,2:].values

    ## go_entity
    go_entity = kg[kg['relation'] == 'Pro_GO']['tail'].unique()
    go_entity_df = pd.DataFrame(go_entity,columns=['entity'])
    go_entity_df['code'] = range(len(go_entity_df))

    ## pathway_entity
    pathway_entity = kg[kg['relation'] == 'PATHWAY_Protein']['head'].unique()
    pathway_entity_df = pd.DataFrame(pathway_entity,columns=['entity'])
    pathway_entity_df['code'] = range(len(pathway_entity_df))

    ### pro_entity_process
    pro_entity = pd.concat([
        kg[kg['relation']=='PPI']['head'],kg[kg['relation']=='PPI']['tail'],
        kg[kg['relation']=='PATHWAY_Protein']['tail'],
        kg[kg['relation']=='Pro_GO']['head'],
        kg[kg['relation']=='DPI']['tail'],
        exp_triples['pro_id']
        ]).unique()
    pro_entity_df = pd.DataFrame(pro_entity,columns=['entity'])
    pro_entity_df['code'] = range(len(pro_entity_df))

    entity_df = pd.concat([drug_entity_df,pro_entity_df,go_entity_df,pathway_entity_df])

    kg['head_id'] = pd.merge(kg,entity_df,how='left',left_on='head',right_on='entity')['code'].astype('int')
    kg['tail_id'] = pd.merge(kg,entity_df,how='left',left_on='tail',right_on='entity')['code'].astype('int')

    exp_triples['head_id'] = pd.merge(exp_triples,entity_df,how='left',left_on='pro_id',right_on='entity')['code'].astype('int')
    exp_triples['tail_id'] = pd.merge(exp_triples,pid_entity_df,how='left',left_on='pid',right_on='entity')['code'].astype('int')
    exp_triples_graph['head_id'] = pd.merge(exp_triples_graph,entity_df,how='left',left_on='pro_id',right_on='entity')['code'].astype('int')
    exp_triples_graph['tail_id'] = pd.merge(exp_triples_graph,pid_entity_df,how='left',left_on='pid',right_on='entity')['code'].astype('int')

    dpi_pos = kg[kg['relation'] == 'DPI']
    dpi_neg['head_id'] = pd.merge(dpi_neg,entity_df,how='left',left_on='head',right_on='entity')['code'].astype('int')
    dpi_neg['tail_id'] = pd.merge(dpi_neg,entity_df,how='left',left_on='tail',right_on='entity')['code'].astype('int')

    pro_path_neg_sp['head_id'] = pd.merge(pro_path_neg_sp,entity_df,how='left',left_on='head',right_on='entity')['code'].astype('int')
    pro_path_neg_sp['tail_id'] = pd.merge(pro_path_neg_sp,entity_df,how='left',left_on='tail',right_on='entity')['code'].astype('int')
    pro_path_pos = kg[kg['relation'] == 'PATHWAY_Protein']

    pid_fea = pd.merge(pid_entity_df,exp_input,how='left',left_on='entity',right_on='pid').iloc[:,3:].values.astype('float')
    print(pid_fea.shape)
    sc_process = MinMaxScaler((-1,1))
    pid_fea_sc = sc_process.fit_transform(pid_fea)

    return kg,dls,\
            pid_entity_df,drug_entity_df,pro_entity_df,go_entity_df,pathway_entity_df,drug_fea,dpi_pos,pro_path_pos,pid_fea_sc



def construct_graph(kg,exp_triples_graph,pid_entity_df,drug_entity_df,pro_entity_df,go_entity_df,pathway_entity_df):
    # Construct graph...
    g = dgl.heterograph({
        ('protein', 'ppi', 'protein'): ([], []),
        ('protein','pro_pid','pid'): ([], []),
        ('drug', 'dpi', 'protein'): ([], []),
        ('protein','pro_path','pathway'): ([], []),
        ('protein','pro_go','go'): ([], []),
    })
    g.add_nodes(len(drug_entity_df), ntype='drug')
    g.add_nodes(len(pro_entity_df), ntype='protein')
    g.add_nodes(len(pid_entity_df), ntype='pid')
    g.add_nodes(len(pathway_entity_df), ntype='pathway')
    g.add_nodes(len(go_entity_df), ntype='go')

    g.add_edges(kg[kg['relation'] == 'DPI']['head_id'].values, kg[kg['relation'] == 'DPI']['tail_id'].values, etype='dpi')
    g.add_edges(kg[kg['relation'] == 'PPI']['head_id'].values, kg[kg['relation'] == 'PPI']['tail_id'].values, etype='ppi')
    g.add_edges(kg[kg['relation'] == 'PATHWAY_Protein']['tail_id'].values, kg[kg['relation'] == 'PATHWAY_Protein']['head_id'].values, etype='pro_path')
    g.add_edges(kg[kg['relation'] == 'Pro_GO']['head_id'].values, kg[kg['relation'] == 'Pro_GO']['tail_id'].values, etype='pro_go')

    ### protein-pid edges
    g.add_edges(exp_triples_graph['head_id'].values, exp_triples_graph['tail_id'].values,
                    #{'label': torch.from_numpy(rna_triples['rna'].values).to(torch.float32)},
                    etype='pro_pid')
    return g