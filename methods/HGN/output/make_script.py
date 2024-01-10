import os
import pandas as pd
import shutil
dataset =pd.DataFrame()
for state in range(50):
    os.chdir('/home/linjw/GNNs/HGB/NC/benchmark/data/ERMer')
    os.system(f'python makedata_script.py {state}')

    os.chdir('/home/linjw/GNNs/HGB/NC/benchmark/methods/baseline')
    os.system(f'python run_new.py --dataset ERM --feats-type 2 --run {state} --weight {float(state + 1)}')

    os.chdir('/home/linjw/GNNs/HGB/NC/benchmark/methods/baseline/output/save')
    output = pd.read_table(f"/home/linjw/GNNs/HGB/NC/benchmark/methods/baseline/output/save/ERM_{state}.txt",header=None)
    label = pd.read_table('/home/linjw/GNNs/HGB/NC/benchmark/data/ERM/label.dat.test',header=None)
    
    source_path = '/home/linjw/GNNs/HGB/NC/benchmark/data/ERM/label.dat.test'
    destination_directory = '/home/linjw/GNNs/HGB/NC/benchmark/methods/baseline/output/save/'
    new_filename = f'label_{state}.dat.test'
    shutil.copy(source_path, destination_directory + new_filename)

    label = label.sort_values(0)
    output.columns = ['Idx','Ept','GroudTruth','Predict']
    output['GroudTruth'] = label[3].values
    output = output[['Idx','GroudTruth','Predict']]
    test_node = pd.read_csv("/home/linjw/GNNs/HGB/NC/benchmark/data/ERMer/test_node.csv")
    target = output[(output['GroudTruth']!=1) & (output['Predict']==1)]
    out = test_node[test_node['nid'].isin(target.Idx.values)]
    dataset = pd.concat([dataset,out])
    # dataset = dataset.drop_duplicates()
    dataset.index = range(len(dataset))
    dataset.to_excel("select_output.xlsx")