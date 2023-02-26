import torch
import dgl
from dgl.data.utils import load_labels
from torch.utils.data import Dataset, DataLoader
import os

class GNodeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)

    def __getitem__(self, item):
        file_name = self.files[item]
        file_name = os.path.join(self.root_dir, file_name)
        (load_g, ),_  = dgl.load_graphs(file_name)
        graph_sizes = load_labels(file_name)
        return load_g, graph_sizes

    def __len__(self):
        return len(self.files)


def collate(samples):
    assert len(samples)==1
    G, graph_sizes = samples[0]
    return G, graph_sizes['graph_sizes'], G.ndata['label']

def GNodeDataloader(path, shuffle=False, num_workers=0):
    dataset = GNodeDataset(path)
    return DataLoader(dataset=dataset,batch_size=1,shuffle=shuffle,num_workers=num_workers, collate_fn=collate)



if __name__=='__main__':
    DEBUG = 0
    if DEBUG==0:
        path = '../datasets/eval'
        for x in GNodeDataloader(path):
            print(x)
            # break



