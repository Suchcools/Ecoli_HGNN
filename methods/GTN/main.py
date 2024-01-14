import sys
sys.path.append('../../')
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score,accuracy
from scripts.data_loader import data_loader
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix
import os

def load_data(args):
    dataset, full, feats_type = args.dataset, args.full, args.feats_type
    drop_feat = []
    if dataset == 'DBLP':
        if feats_type == 1:
            drop_feat = [0,1,3]
        elif feats_type == 2:
            drop_feat = [0]
    dl = data_loader('../../data/'+dataset)
    if dataset == 'DBLP' and not full:
        dl.get_sub_graph([0,1,3])
    if dataset == 'ACM' and not full:
        dl.get_sub_graph([0,1,2])
    edges = list(dl.links['data'].values())
    node_features = []
    w_ins = []
    for k in dl.nodes['attr']:
        th = dl.nodes['attr'][k]
        if th is None or k in drop_feat:
            cnt = dl.nodes['count'][k]
            node_features.append(sp.eye(cnt))
            w_ins.append(cnt)
        else:
            node_features.append(th)
            w_ins.append(th.shape[1])
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_label = labels[train_idx]
    val_label = labels[val_idx]
    train_label = list(zip(train_idx.tolist(), train_label.tolist()))
    val_label = list(zip(val_idx.tolist(), val_label.tolist()))
    labels = [train_label, val_label, test_idx.tolist()]
    return node_features, edges, labels, dl, w_ins

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)
# weight 12  num_channels 5 lr 5e-3
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ERM",
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=5,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--full', type=bool, default=False)
    parser.add_argument('--feats-type', type=int, default=0)
    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_features, edges, labels, dl, w_ins = load_data(args)
    num_nodes = edges[0].shape[0]

    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    node_features = [mat2tensor(x) for x in node_features] 
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])).type(torch.LongTensor)
    num_classes = torch.max(train_target).item()+1
    final_f1 = 0
    for l in range(1):
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_ins = w_ins, 
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm).to(device)
 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        
        # Lists to store training and validation losses
        train_losses = []
        val_losses = []

        # Lists to store training and validation F1 scores
        train_f1_scores = []
        val_f1_scores = []

        # Lists to store training and validation confusion matrices
        train_conf_matrices = []
        val_conf_matrices = []
        
        
        for i in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ',i+1)
            model.zero_grad()
            model.train()
            loss,y_train,Ws = model(A.to(device), [x.to(device) for x in node_features], train_node.to(device), train_target.to(device))
            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach().cpu(),dim=1), train_target.detach().cpu(), num_classes=num_classes)).cpu().numpy()
            loss.backward() 
            
            optimizer.step()
            model.eval()
            print('Train - Loss: {}, Macro_F1: {} , TF_F1: {}'.format(loss.detach().cpu().numpy(), train_f1, f1_score(torch.argmax(y_train.detach().cpu(),dim=1), train_target.detach().cpu(), num_classes)[1].item()))
            
            # Save training loss and F1 score
            train_losses.append(loss.item())
            train_f1_scores.append(train_f1)
            
            # Save training confusion matrix
            train_conf_matrix = confusion_matrix(
                train_target.detach().cpu(), torch.argmax(y_train.detach().cpu(), dim=1)
            )
            train_conf_matrices.append(train_conf_matrix)
                    
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_ = model.forward(A.to(device), [x.to(device) for x in node_features], valid_node.to(device), valid_target.to(device))
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid.detach().cpu(),dim=1), valid_target.detach().cpu(), num_classes=num_classes)).cpu().numpy()
                print('Valid - Loss: {}, Macro_F1: {}, TF_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1, f1_score(torch.argmax(y_valid.detach().cpu(),dim=1), valid_target.detach().cpu(), num_classes)[1].item()))
                test_loss, y_test,W = model.forward(A.to(device), [x.to(device) for x in node_features], test_node.to(device), None)
                pred = y_test.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                valid_acc = accuracy(torch.argmax(y_valid.detach().cpu(),dim=1), valid_target.detach().cpu())
                
                # Save validation loss and F1 score
                val_losses.append(val_loss.item())
                val_f1_scores.append(val_f1)

                # Save validation confusion matrix
                val_conf_matrix = confusion_matrix(
                    valid_target.detach().cpu(), torch.argmax(y_valid.detach().cpu(), dim=1)
                )
                val_conf_matrices.append(val_conf_matrix)
                print(val_conf_matrix)

                
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                
                
            # Create output directory if it does not exist
            output_dir = 'output'
            
            # Convert lists to numpy arrays
            train_losses_array = np.array(train_losses)
            val_losses_array = np.array(val_losses)
            train_f1_array = np.array(train_f1_scores)
            val_f1_array = np.array(val_f1_scores)

            # Save arrays as .npy files in the output directory
            np.save(os.path.join(output_dir, 'train_losses.npy'), train_losses_array)
            np.save(os.path.join(output_dir, 'val_losses.npy'), val_losses_array)
            np.save(os.path.join(output_dir, 'train_f1_scores.npy'), train_f1_array)
            np.save(os.path.join(output_dir, 'val_f1_scores.npy'), val_f1_array)

            # Save confusion matrices as .npy files in the output directory
            np.save(os.path.join(output_dir, 'train_conf_matrices.npy'), train_conf_matrices)
            np.save(os.path.join(output_dir, 'val_conf_matrices.npy'), val_conf_matrices)

        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
