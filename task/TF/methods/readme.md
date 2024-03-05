FM-HGN python run.py
GTN python main.py --dataset ERM --num_layers 1 --adaptive_lr false --num_channels 10 --norm False
GAT python run.py --dataset ERM --model-type gat --feats-type 2
GCN python run.py --dataset ERM --model-type gcn --weight-decay 1e-6 --lr 1e-3 --feats-type=0