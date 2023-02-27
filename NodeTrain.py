import torch.optim
import torch
import sys
from NodeModel import NodeGNN
from data_process.dataset import GNodeDataset,GNodeDataloader
import torch.nn as nn

train_path = 'datasets/train'
eval_path = 'datasets/test'
loss_func = torch.nn.CrossEntropyLoss()
cuda = True
device = torch.device(f'cuda:2')

# torch.manual_seed(3407)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()

def train():
    model = NodeGNN(98, 36, 64, 4, 4, 'cuda:2')
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else :
            weight_p += [p]
    model.apply(initialize_weights)

    if cuda:
        model = model.to(device)
    data_loader = GNodeDataloader(train_path, shuffle=True)
    data_loader_eval = GNodeDataloader(eval_path, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam([{'params':weight_p, 'weight_decay':0.00001},
    #                               {'params':bias_p, 'weight_decay':0}], lr = 0.001)
    
    step = 0
    every_print = 100
    every_eval = 1000
    best_test_acc = 0

    for epoch in range(500):
        print(f'epoch {epoch} starts')
        train_accuracy_list = []
        for x in data_loader:
            step += 1

            g_batch, graph_sizes, label_batch = x
            if cuda:
                g_batch, graph_sizes, label_batch = g_batch.to(device), graph_sizes.to(device), label_batch.to(device)
            metric = model(g_batch,g_batch.ndata['feat'],g_batch.edata['feat'])
            tup = metric.split(tuple(graph_sizes))
            label = label_batch.split(tuple(graph_sizes))
            T = 0
            for i in range(len(tup)):
                pred = torch.argmax(tup[i],1)
                if(torch.equal(pred.int(), label[i].int())):
                    T += 1
            train_accuracy = T/len(graph_sizes)
            train_accuracy_list.append(train_accuracy)
            
            optimizer.zero_grad()
            loss = loss_func(metric, label_batch.long())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if(step%every_print==0):
                print(f'step {step} train loss is: {loss}', flush=True)

            if(step%every_eval==0):
                model.eval()
                eval_accuracy_list = []
                for x in data_loader_eval:
                    g_batch, graph_sizes, label_batch = x
                    if cuda:
                        g_batch, graph_sizes, label_batch = g_batch.to(device), graph_sizes.to(device), label_batch.to(device)

                    metric = model(g_batch, g_batch.ndata['feat'], g_batch.edata['feat'])
                    tup = metric.split(tuple(graph_sizes))
                    label = label_batch.split(tuple(graph_sizes))
                    T = 0
                    for i in range(len(tup)):
                        pred = torch.argmax(tup[i],1)
                        if(torch.equal(pred.int(), label[i].int())):
                            T += 1

                    eval_accuracy = T/len(graph_sizes)
                    eval_accuracy_list.append(eval_accuracy)
                print('------------------------------')
                cur_eval_acc = sum(eval_accuracy_list)/len(eval_accuracy_list)
                print(f'step {step} eval accuracy is: {cur_eval_acc}', flush=True)
                if cur_eval_acc >= best_test_acc:
                    best_test_acc = cur_eval_acc
                    torch.save(model, 'NodeLevelBestModel.pt')
                print('------------------------------',flush=True)
                model.train()

        print('***************************')
        print(f'epoch {epoch} train accuracy is: {sum(train_accuracy_list)/len(train_accuracy_list)}', flush=True )
        print('***************************', flush=True)


    print('Final: ')
    model.eval()
    eval_accuracy_list = []
    for x in data_loader_eval:
        g_batch, graph_sizes, label_batch = x
        if cuda:
            g_batch, graph_sizes, label_batch = g_batch.to(device), graph_sizes.to(device), label_batch.to(device)

        metric = model(g_batch, g_batch.ndata['feat'], g_batch.edata['feat'])
        tup = metric.split(tuple(graph_sizes))
        label = label_batch.split(tuple(graph_sizes))
        T = 0
        for i in range(len(tup)):
            pred = torch.argmax(tup[i],1)
            if(torch.equal(pred.int(), label[i].int())):
                T += 1

        eval_accuracy = T/len(graph_sizes)
        eval_accuracy_list.append(eval_accuracy)
    
    cur_eval_acc = sum(eval_accuracy_list)/len(eval_accuracy_list)
    if cur_eval_acc >= best_test_acc:
        best_test_acc = cur_eval_acc
        torch.save(model, 'NodeLevelBestModel.pt')
    print('------------------------------')
    print(f'best eval accuracy is: {best_test_acc}', flush=True)
    print('------------------------------')
    model.train()


if __name__ =='__main__':
    train()
    


