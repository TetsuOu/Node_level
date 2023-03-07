import torch.optim
import torch
import sys
from NodeModel import NodeGNN
from data_process.dataset import GNodeDataset,GNodeDataloader
import torch.nn as nn

from sklearn.metrics import f1_score

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
eval_path = 'datasets/test'
data_loader_eval = GNodeDataloader(eval_path, shuffle=False)

model = torch.load('NodeLevelBestModel.pt', map_location=torch.device(device))

eval_accuracy_list = []
y_true = []
y_pred = []
for x in data_loader_eval:
    g_batch, graph_sizes, label_batch = x
    if torch.cuda.is_available():
        g_batch, graph_sizes, label_batch = g_batch.to(device), graph_sizes.to(device), label_batch.to(device)
    metric = model(g_batch, g_batch.ndata['feat'], g_batch.edata['feat'])
    tup = metric.split(tuple(graph_sizes))
    label = label_batch.split(tuple(graph_sizes))
    T = 0
    all_one = 0
    true_one = 0
    for i in range(len(tup)):
        # if torch.max(label[i]) != 1.0:
        #     continue
        pred = torch.argmax(tup[i], 1)
        # if (torch.equal(pred.int(), label[i].int()) and torch.max(label[i]) == 1.0):
        #     true_one += 1
        if (torch.equal(pred.int(), label[i].int())):
            T += 1
        
        y_true.extend(pred.cpu())
        y_pred.extend(label[i].int().cpu())

    eval_accuracy = T / len(graph_sizes)
    eval_accuracy_list.append(eval_accuracy)
    # print(x)
    

print('f1 score: ', f1_score(y_true, y_pred))
print(sum(eval_accuracy_list)/len(eval_accuracy_list))
