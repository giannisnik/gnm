import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score

from models import MLP, GNM
from utils import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gnm', choices=['mlp', 'gnm'], help='Model (GNM or MLP)')
parser.add_argument('--dataset', default='wine', choices=['wine', 'yeast', 'wireless', 'noisy_moons'], help='Dataset name')
parser.add_argument('--cv', type=int, default=10, help='Number of folds of cross-validation')
parser.add_argument('--n-nodes', type=int, default=50, help='Number of nodes of the graph of GNM')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--n-layers', type=int, default=3, help='Number of layers of GNM and MLP models')
parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size (only for MLP)')
parser.add_argument('--trainable-features', action='store_true', default=False, help='Whether to learn initial node features of non-input nodes (only for GNM)')
args = parser.parse_args()

X, y = load_dataset(args.dataset)
n = X.shape[0]
m = X.shape[1]

if args.dataset in {'wine'}:
    task = 'regression'
    loss_function = nn.MSELoss()
    if y.ndim == 1:
        output_dim = 1
    else:
        output_dim = y.shape[1]
else:
    output_dim = np.unique(y).size
    if output_dim == 2:
        task = 'binary_classification'
        loss_function = nn.BCEWithLogitsLoss()
        output_dim = 1
    else:
        task = 'multiclass_classification'
        loss_function = nn.CrossEntropyLoss()

print('--------------------')
print('Number of samples:', n)
print('Number of features:', m)
print('Number of outputs:', output_dim)

X = torch.from_numpy(X).float()
if task == 'multiclass_classification':
    y = torch.from_numpy(y).long()
else:
    y = torch.from_numpy(y).float()

if args.model == 'gnm':
    assert args.n_nodes > m+output_dim 
    input_idx = list(range(m))
    output_idx = list(range(args.n_nodes-output_dim, args.n_nodes))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(epoch, loader):
    model.train()
    loss_all = 0

    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(x), y)
        loss.backward()
        loss_all += y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)

def val(loader):
    model.eval()
    loss_all = 0

    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        loss = loss_function(model(x), y)
        loss_all += y.size(0) * loss.item()
    return loss_all / len(loader.dataset)

def test(loader):
    model.eval()
    eval_metrics = dict()

    y_pred = list()
    y_test = list()
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        if task == 'binary_classification':
            pred = torch.where(pred > 0.5, 1.0, 0.0)
        elif task == 'multiclass_classification':
            pred = pred.max(1)[1]

        y_pred.append(pred.detach().cpu())
        y_test.append(y.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_test = torch.cat(y_test, dim=0).numpy()
    if task == 'regression':
        eval_metrics['mse'] = mean_squared_error(y_test, y_pred) 
        eval_metrics['r2'] = r2_score(y_test, y_pred) 
    else:
        eval_metrics['acc'] = accuracy_score(y_test, y_pred)
        eval_metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro')
        eval_metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')

    return eval_metrics


kf = KFold(n_splits=args.cv, shuffle=True)
evals = list()
for it, (train_index, test_index) in enumerate(kf.split(X)):
    train_index, val_index = train_test_split(train_index, test_size=0.1)
    X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    if args.model == 'gnm':
        model = GNM(args.n_nodes, args.n_layers, input_idx, output_idx, args.trainable_features, args.dropout, device).to(device)        
    else:
        model = MLP(m, args.hidden_dim, output_dim, args.n_layers, args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('--------------------')
    print('Number of parameters:', total_params)
    print('--------------------')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print('\nSplit:', it)
    print('--------------------')

    best_val_loss, test_acc = np.inf, 0
    for epoch in range(1, args.epochs+1):
        train_loss = train(epoch, train_loader)
        val_loss = val(val_loader)
        if best_val_loss >= val_loss:
            eval_metrics = test(test_loader)
            best_val_loss = val_loss
        if epoch % 50 == 0:
            if task == 'regression':
                print('Epoch: {:03d}, Train Loss: {:.7f}, '
                        'Val Loss: {:.7f}, Test Loss: {:.7f}'.format(
                        epoch, train_loss, val_loss, eval_metrics['mse']))
            else:
                print('Epoch: {:03d}, Train Loss: {:.7f}, '
                        'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                        epoch, train_loss, val_loss, eval_metrics['acc']))
    
    if task == 'regression':
        evals.append(eval_metrics['mse'])
    else:
        evals.append(eval_metrics['acc'])

print('\nFinal Result')
print('--------------------')
print('Mean: {:7f}, Std: {:7f}'.format(np.mean(evals), np.std(evals)))