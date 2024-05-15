import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import os
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

for random_state in tqdm(range(101,150)):
    path = f"../data/INS/{random_state}"
    os.makedirs(path, exist_ok=True)

    nums_col = ["Len:Int","LeftEnd:Int","RightEnd:Int","Cs:Float","MW:Float","Orthology_org:String","ProteinOntology:String"]
    cls_col = ["Ori:String","Operon:String","Uber_Operon:String"]
    other_col = ["ENZYME:String","GO:String[]"]
    node_info = pd.read_csv('../data/Process/Gene_Entity_v2.csv')
    gene = node_info[['id','amino_acid_seq:String','~label']]
    gene = gene[gene['amino_acid_seq:String']!='none']
    def get_term(go_term):
        try:
            name = list(set([x['go_term_name'] for x in eval(go_term).values()]))
            belong = list(set([x['go_belong'] for x in eval(go_term).values()]))
            return name ,belong
        except:
            return "",""
        
    # other 
    mlb = MultiLabelBinarizer()
    term_encode = mlb.fit_transform([x[0] for x in node_info["GO:String[]"].apply(get_term).values])
    mlb = MultiLabelBinarizer()
    go_belong_encode = mlb.fit_transform([x[1] for x in node_info["GO:String[]"].apply(get_term).values])
    ec_num = node_info["ENZYME:String"].fillna("-1.").apply(lambda x:x.split(".")[0]).values.astype(int).reshape(-1,1)

    # cls
    label_encoder = LabelEncoder()
    cls_encoded = node_info[cls_col].apply(lambda col: label_encoder.fit_transform(col))

    df_float = node_info[nums_col].fillna(-1)
    df_float = df_float.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(-1).astype(float)

    node_feat = np.concatenate((term_encode, go_belong_encode, ec_num, cls_encoded.values, df_float.values), axis=1)
    # MinMaxScaler
    scaler = MinMaxScaler()
    node_feat = scaler.fit_transform(node_feat)
    node_feat_dict = dict(zip(node_info.id.values,node_feat))
    node_feat.shape


    gene = pd.read_csv('../data/Process/Gene_Entity_v2.csv')[['id','amino_acid_seq:String','~label']]
    gene = gene[gene['amino_acid_seq:String']!='none']
    metabolite = pd.read_csv('../data/Process/Metabolite_Entity_v5.csv')[['id','label']]
    pathway = pd.read_csv('../data/Process/Pathway_Entity_v3.csv')[['id','label']]
    reaction = pd.read_csv('../data/Process/Reaction_Entity_v5.csv')[['id','label']]
    node = pd.concat([gene,metabolite,pathway,reaction])
    # node = node[node["~label"]!="Gene;SIGMA"]
    node = node.drop_duplicates(subset='id')
    node.index = range(len(node)) 
    node.label=node.label.fillna('Gene')
    node['~label'] = node['~label'].replace(['Gene','Gene;TF','Gene;SIGMA'],['Non-TF','TF','TF'])
    node['~label'].value_counts()


    edge = pd.read_csv('../data/Process/Edge20220217_V3.csv')
    edge = edge[['eid','source','target','attribute','label']]
    edge.eid = edge.eid.apply(lambda x:x.replace("Edge_",""))
    edge_order = ['TFGI','CPI','sRGI','GRI','MRI','RMI','SFGI','PPI','RPI']
    edge['label'] = edge['label'].replace(edge_order,[0,1,8,2,3,4,5,6,7]) # 2 1 3 5 4 7 0 6
    edge = edge.sort_values('label')
    edge = edge.fillna("")
    label_encoder = LabelEncoder()
    edge["attribute"] = label_encoder.fit_transform(edge["attribute"].values)

    # 对节点数据进行编号
    node['nid'] = range(len(node))
    # 创建一个字典，将节点的id映射到nid
    id_to_nid = dict(zip(node['id'], node['nid']))

    # 替换边关系数据中的source和target列为对应的nid值
    edge['source'] = edge['source'].map(id_to_nid)
    edge['target'] = edge['target'].map(id_to_nid)
    edge = edge.dropna()
    edge.index = range(len(edge))
    edge = edge.astype(int)
    edge.eid = range(len(edge))

    edge_nodes = set(edge['source']).union(set(edge['target']))
    node = node[node['nid'].isin(edge_nodes)]
    # 按照给定索引顺序创建一个新的DataFrame
    index_order = ['Gene', 'Reaction', 'Metabolite', 'Pathway']
    node = node[node['label'].isin(index_order)].sort_values(by=['label'], key=lambda x: x.map({label: i for i, label in enumerate(index_order)}))

    node_mapping = {}
    new_nid = 0
    for index, row in node.iterrows():
        node_mapping[row['nid']] = new_nid
        new_nid += 1
    edge['source'] = edge['source'].map(node_mapping)
    edge['target'] = edge['target'].map(node_mapping)
    node['nid'] = node['nid'].map(node_mapping)

    # 使用 dropna() 方法删除含有 NA 的行并返回新的 DataFrame
    df_without_na = edge.dropna()

    # 或者使用布尔索引筛选出含有 NA 的行而不删除它们
    df_with_na = edge[edge.isna().any(axis=1)]
    edge = edge.dropna()

    label_df = node[node['~label'].notna()]
    label_df['~label'] = label_df['~label'].replace(['Non-TF','TF'],[0,1])

    # 随机打乱数据行的顺序
    # label_df = label_df.sample(frac=1, random_state=2020).reset_index(drop=True)
    label_df = label_df.sample(frac=1, random_state=1).reset_index(drop=True)


    # 指定测试集所占的比例
    test_ratio = 0.3
    train_data, test_data = train_test_split(label_df[["id","~label","label","nid"]], test_size=test_ratio, random_state=random_state)

    # 将数据写入 label.dat 和 label.dat.test 文件
    with open(f'{path}/label.dat', 'w', encoding='utf-8') as label_file:
        for index, row in train_data.iterrows():
            label_file.write('{}\t\t{}\t{}\n'.format(row['nid'], 0, row['~label']))

    with open(f'{path}/label.dat.test', 'w', encoding='utf-8') as test_label_file:
        for index, row in test_data.iterrows():
            test_label_file.write('{}\t\t{}\t{}\n'.format(row['nid'], 0, row['~label']))
            
    train_data.index = range(len(train_data))
    train_data.to_csv(f"{path}/train_node.csv",index=False)
    test_data.to_csv(f"{path}/test_node.csv",index=False)

    def pandas_to_fasta(dataframe, id_column, sequence_column, output_file):
        with open(output_file, 'w') as f:
            for index, row in dataframe.iterrows():
                identifier = row[id_column]
                sequence = row[sequence_column]
                f.write(f">{identifier}\n{sequence}\n")

    # 假设你的Pandas DataFrame 名称为df，其中'id'列包含序列标识符，'sequence'列包含序列数据
    pandas_to_fasta(node[['nid','amino_acid_seq:String']], 'nid', 'amino_acid_seq:String', 'input.fasta')

    edge['source']=edge['source'].astype(int)
    edge['target']=edge['target'].astype(int)
    edge = edge.drop_duplicates(subset=['source','target'], keep='first', inplace=False)
    edge = edge.sort_values(by=['label','source','target'])

    nodedat = open(f'{path}/link.dat', 'w', encoding='utf-8')
    for index,row in edge.iterrows():
        nodedat.write('{}\t{}\t{}\t{}\t{}\n'.format(row['source'], row['target'], row['label'], row['attribute'] ,1.0))
    nodedat.close()

    import os
    import torch
    def find_image_file(source_path, file_lst):
        """
        递归寻找 文件夹以及子目录的 图片文件。
        :param source_path: 源文件夹路径
        :param file_lst: 输出 文件路径列表
        :return:
        """
        image_ext = ['.pt']
        for dir_or_file in os.listdir(source_path):
            file_path = os.path.join(source_path, dir_or_file)
            if os.path.isfile(file_path):  # 判断是否为文件
                file_name_ext = os.path.splitext(os.path.basename(file_path))  # 文件名与后缀
                if len(file_name_ext) < 2:
                    continue
                if file_name_ext[1] in image_ext:  # 后缀在后缀列表中
                    file_lst.append(file_path)
                else:
                    continue
            elif os.path.isdir(file_path):  # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
                find_image_file(file_path, file_lst)
            else:
                print('文件夹没有环境' + os.path.basename(file_path))
    env_path_list=[]
    find_image_file('../../../data/ESM_Protein_Embedding',env_path_list)
    label = [x.split('/')[-1].replace('.pt','') for x in env_path_list]
    class_labels = ['dandelion','daisy','sunflower']
    rawdata = pd.DataFrame([env_path_list,label],index=['path','label']).T
    rawdata

    protein_embedding = rawdata.path.apply(lambda x :torch.load(x)['mean_representations'][0].tolist())
    rawdata['embedding'] = protein_embedding

    import numpy as np
    # 创建一个字典，用于存储label和embedding的对应关系
    embedding_dict = dict(zip(rawdata['label'], rawdata['embedding']))

    # 定义一个函数，用于根据label获取对应的embedding
    def get_embedding(label):
        try:
            protein_emb = embedding_dict[label]
            node_feat_emb = node_feat_dict.get(label).tolist()
            if node_feat_emb!= None:
                protein_emb.extend(node_feat_emb)
            else:
                protein_emb.extend([0]*node_feat.shape[1])
            return protein_emb
        except:
            return np.nan

    # 将embedding补全到node数据中
    node['embedding'] = node['id'].apply(get_embedding)

    nodedat = open(f'{path}/node.dat', 'w', encoding='utf-8')
    node['label'] = node['label'].replace(index_order,[0,1,2,3]) # 都变成0
    for index,row in node.iterrows():
        try:
            nodedat.write('{}\t{}\t{}\t{}\n'.format(row['nid'], row['amino_acid_seq:String'], row['label'], ','.join([str(item) for item in row['embedding']])))
        except:
            nodedat.write('{}\t{}\t{}\n'.format(row['nid'], row['amino_acid_seq:String'], row['label']))
    nodedat.close()

    import json
    # edge_order = ['TFGI','CPI','sRGI','GRI','MRI','RMI','SFGI','PPI','RPI']
    # edge['label'] = edge['label'].replace(edge_order,[0,1,2,3,4,5,6,7,8]) # 2 1 3 5 4 7 0 6
    info = {
    'node.dat': {'node type': {0: 'Gene', 1: 'Reaction', 2: 'Metabolite', 3: 'Pathway'}},

    'label.dat': {'node type': {0: {0: 'Non-TF', 1: 'TF'}}},

    'link.dat': {
            "link type": {
                "0": {
                    "start": 0,
                    "end": 0,
                    "meaning": "TFGI"
                },
                "1": {
                    "start": 2,
                    "end": 0,
                    "meaning": "CPI"
                },
                "2": {
                    "start": 0,
                    "end": 1,
                    "meaning": "GRI"
                },
                "3": {
                    "start": 2,
                    "end": 1,
                    "meaning": "MRI"
                },
                "4": {
                    "start": 1,
                    "end": 2,
                    "meaning": "RMI"
                },
                "5": {
                    "start": 0,
                    "end": 0,
                    "meaning": "SFGI"
                },
                "6": {
                    "start": 0,
                    "end": 0,
                    "meaning": "PPI"
                },
                "7": {
                    "start": 1,
                    "end": 3,
                    "meaning": "RPI"
                }
            }}
    }

    with open(f'{path}/info.dat', 'w', encoding='utf-8') as info_file:
        json.dump(info, info_file, indent=4)
    
    