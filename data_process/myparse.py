import pandas as pd
import os
import sys

import torch
import argparse
sys.path.append('..')
from seq_graph_retro.utils.parse import get_reaction_info
from seq_graph_retro.utils.chem import get_mol, apply_edits_to_mol
from dgl_graph.utils import SmileToDGLGraph
import rdkit.Chem as Chem
import dgl

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
args = parser.parse_args()

DATA_DIR = '../datasets'

file_name = f'canonicalized_{args.mode}.csv'

data_df = pd.read_csv(os.path.join(DATA_DIR, file_name))

BATCH_SIZE = 64
batch_cnt = 0
dgl_batch = []
num = 0

for idx in range(len(data_df)):
    element = data_df.iloc[idx]
    rxn_smi = element['reactants>reagents>production']

    try:
        reaction_info = get_reaction_info(rxn_smi, kekulize=True, use_h_labels=True)
    except Exception as e:
        continue

    core_edits = reaction_info.core_edits

    if len(core_edits)>1 :
        print(f'{idx} is multiedit, skip')
        continue

    r, p = rxn_smi.split(">>")
    products = get_mol(p)

    if (products.GetNumBonds() <= 1):
        print(f'Product has 0 or 1 bonds, Skipping reaction {idx}')
        print()
        sys.stdout.flush()
        continue

    if (products is None) or (products.GetNumAtoms() <= 1):
        print(f"Product has 0 or 1 atoms, Skipping reaction {idx}")
        print()
        sys.stdout.flush()
        continue

    reactants = get_mol(r)
    if (reactants is None) or (reactants.GetNumAtoms() <= 1):
        print(f"Reactant has 0 or 1 atoms, Skipping reaction {idx}")
        print()
        sys.stdout.flush()
        continue

    products_dgl = SmileToDGLGraph(Chem.MolToSmiles(products))

    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in products.GetAtoms()}
    core_idx = torch.tensor([p_amap_idx[x] for x in reaction_info.core])
    label = torch.zeros(products_dgl.num_nodes())

    for core in core_edits:
        if(float(core_edits[0].split(':')[1]) == 0.0):
            label[p_amap_idx[int(core.split(':')[0])]] = 1
        else:
            label[p_amap_idx[int(core.split(':')[0])]] = 2
            label[p_amap_idx[int(core.split(':')[1])]] = 2

    products_dgl.ndata['label'] = label
    dgl_batch.append(products_dgl)
    num += 1
    if num%BATCH_SIZE==0:
        dgl_G = dgl.batch(dgl_batch)
        batch_info = {'graph_sizes': dgl_G.batch_num_nodes()}
        dgl.save_graphs(f'{DATA_DIR}/{args.mode}/graph_{batch_cnt}.dgl', dgl_G, batch_info)
        batch_cnt += 1
        dgl_batch.clear()
        if(batch_cnt%100==0):
            print(f'Now {batch_cnt} batches !')

    # print('hello world')

if len(dgl_batch)>0:
    dgl_G = dgl.batch(dgl_batch)
    batch_info = {'graph_sizes': dgl_G.batch_num_nodes()}
    dgl.save_graphs(f'{DATA_DIR}/{args.mode}/graph_{batch_cnt}.dgl', dgl_G, batch_info)
    batch_cnt += 1
    dgl_batch.clear()

print(f'All {batch_cnt} batches !')










