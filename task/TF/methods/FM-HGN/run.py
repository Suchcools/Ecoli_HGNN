import sys
sys.path.append('../../')
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from utils.focalloss import FocalLoss, DSCLoss
from sklearn.metrics import confusion_matrix
from GNN import myGAT
import dgl

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

def run_model(args):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])
                
    type2emb = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            type2emb[(u,v)] = dl.links['type'][k][u, v]
    for i in range(dl.nodes['total']):
        if (i,i) not in type2emb:
            type2emb[(i,i)] = dl.links['type'][k][i, i]
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in type2emb:
                type2emb[(v,u)] = dl.links['type'][k][v, u]             
                
    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    e_type = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
        e_type.append(type2emb[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    e_type = torch.tensor(e_type, dtype=torch.long).to(device)
    
    # Lists to store epoch data
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    times = []
    
    for _ in range(args.repeat):
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset.replace("/",""), args.num_layers))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            logits = net(features_list, e_feat, e_type)
            logp = F.log_softmax(logits, 1)
            # train_loss = F.nll_loss(logp[train_idx], labels[train_idx])
            train_loss = F.cross_entropy(logp[train_idx], labels[train_idx].argmax(axis=1), weight=torch.tensor([1.0, args.weight]).cuda())
            # train_loss = FocalLoss()(logp[train_idx], labels[train_idx])
            # train_loss = DSCLoss()(logp[train_idx], labels[train_idx])
            
            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()
            
            # Calculate training accuracy
            train_pred = logp[train_idx].argmax(dim=1)
            train_correct = train_pred.eq(labels[train_idx].argmax(axis=1)).sum().item()
            train_accuracy = train_correct / len(train_idx)

            # Record training info
            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy)
            times.append(t_end-t_start)
            
            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Train_Acc: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), train_accuracy, t_end-t_start))


            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list, e_feat, e_type)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx].argmax(axis=1))

                # Calculate validation accuracy
                val_pred = logp[val_idx].argmax(dim=1)
                val_correct = val_pred.eq(labels[val_idx].argmax(axis=1)).sum().item()
                val_accuracy = val_correct / len(val_idx)
            t_end = time.time()
            
            # Record validation info
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy)

            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Val_Acc: {:.4f} | Time(s) {:.4f}'.format(epoch, val_loss.item(), val_accuracy, t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
            
        # Save all epoch data to an npz file after training is complete
        np.savez(f"../../output/{args.name}_info.npz", 
                train_losses=np.array(train_losses), 
                train_accuracies=np.array(train_accuracies), 
                val_losses=np.array(val_losses), 
                val_accuracies=np.array(val_accuracies), 
                times=np.array(times))

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset.replace("/",""), args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(features_list, e_feat, e_type)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            # dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"output/{args.dataset.replace("/","")}_{args.run}.txt")
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"output/1.txt")
            pred = onehot[pred]
            print(dl.evaluate(pred))
            y_true = dl.labels_test['data'][dl.labels_test['mask']].argmax(axis=1)
            prob = test_logits.cpu().numpy()
            y_pred = np.argmax(prob, axis=1)
            save_path = int(args.dataset.split("/")[-1])
            np.savez(f"../../output/instance/{save_path}.npz", label=y_true, prob=prob)
        
            
            # Calculate the classification report
            report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            print(pd.DataFrame(report))
            # Calculate the confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            # Print the confusion matrix
            print(conf_matrix)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='FM-HGN testing for the ERMer dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--weight', type=float, default=5)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str, default="INS/1")
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--name', type=str, default=f"1")
    
    args = ap.parse_args()
    os.makedirs('checkpoint', exist_ok=True)
    run_model(args)