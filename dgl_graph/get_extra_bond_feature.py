import numpy as np
from rdkit import Chem
from typing import Set, Any, List, Union
import bisect

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

BOND_LEN_DICT = {'BR-BR': 228, 'C-B': 156, 'C-BR': 194, 'C-C': 154, 'C--C': 134, 'C---C': 120, 'C-CL': 177,
'C-F': 135, 'C-H': 109, 'C-I': 214, 'C-N': 147, 'C--N': 129, 'C---N': 116, 'C-O': 143, 'C--O': 120,
 'C-P': 184, 'C-S': 182, 'C--S': 160, 'C-SI': 185, 'CL-CL': 199, 'CS-I': 337, 'F-F': 142, 'H-H': 74,
 'H-BR': 141, 'H-CL': 127, 'H-F': 92, 'H-I': 161, 'I-I': 267, 'K-BR': 282, 'K-CL': 267, 'K-F': 217,
  'K-I': 305, 'LI-CL': 202, 'LI-H': 239, 'LI-I': 238, 'N-H': 101, 'N-N': 145, 'N--N': 125,
  'N---N': 110, 'N-O': 140, 'N--O': 121, 'NA-BR': 250, 'NA-CL': 236, 'NA-F': 193, 'NA-H': 189,
  'NA-I': 271, 'O-H': 96, 'O-O': 148, 'O--O': 121, 'P-BR': 220, 'P-CL': 203, 'P-H': 144, 'P-O': 163,
   'P--O': 150, 'PB-O': 192, 'PB-S': 239, 'RB-BR': 294, 'RB-CL': 279, 'RB-F': 227, 'RB-I': 318,
   'S-H': 134, 'S--O': 143, 'S-S': 207, 'S--S': 149, 'SE-H': 146, 'SE-SE': 232, 'SE--SE': 215,
    'H-B': 119, 'H-SI': 148, 'H-GE': 153, 'H-SN': 170, 'H-AS': 152, 'H-TE': 170, 'B-CL': 175,
    'C-GE': 195, 'C-SN': 216, 'C-PB': 230, 'C---O': 113, 'SI-SI': 233, 'SI-O': 163, 'SI-S': 200,
    'SI-F': 160, 'SI-CL': 202, 'SI-BR': 215, 'SI-I': 243, 'GE-GE': 241, 'GE-F': 168, 'GE-CL': 210,
    'GE-BR': 230, 'SN-CL': 233, 'SN-BR': 250, 'SN-I': 270, 'PB-CL': 242, 'PB-I': 279, 'N-F': 136,
    'N-CL': 175, 'P-P': 221, 'P--S': 186, 'P-F': 154, 'AS-AS': 243, 'AS-O': 178, 'AS-F': 171,
    'AS-CL': 216, 'AS-BR': 233, 'AS-I': 254, 'O-F': 142, 'S-F': 156, 'S-CL': 207, 'I-F': 191,
    'I-CL': 232, 'XE-O': 175, 'XE-F': 195, 'AROMATIC': 144}
BOND_LEN_BOUNDARIES = [-2, 0, 74.0, 114.0, 123.0, 136.0, 143.0, 148.0, 154.0, 162.0, 174.0,
                       184.0, 193.0, 202.0, 215.0, 221.0, 232.0, 239.0, 250.0, 277.0]

BOND_ENERGY_DICT = {'B-F': 613, 'B-O': 536, 'BR-BR': 190, 'C-B': 356, 'C-BR': 285, 'C-C': 346, 'C--C': 256, 'C---C': 233,
 'C-CL': 327, 'C-F': 485, 'C-H': 411, 'C-I': 213, 'C-N': 305, 'C--N': 310, 'C---N': 272, 'C-O': 358,
  'C--O': 441, 'C-P': 264, 'C-S': 272, 'C--S': 301, 'C-SI': 318, 'CL-CL': 240, 'CS-I': 337,
  'F-F': 155, 'H-H': 432, 'H-BR': 362, 'H-CL': 428, 'H-F': 565, 'H-I': 295, 'I-I': 148, 'K-BR': 380,
  'K-CL': 433, 'K-F': 498, 'K-I': 325, 'LI-CL': 469, 'LI-H': 238, 'LI-I': 345, 'N-H': 386, 'N-N': 167,
  'N--N': 251, 'N---N': 524, 'N-O': 201, 'N--O': 406, 'NA-BR': 367, 'NA-CL': 412, 'NA-F': 519, 'NA-H': 186,
   'NA-I': 304, 'O-H': 459, 'O-O': 142, 'O--O': 352, 'P-BR': 264, 'P-CL': 326, 'P-H': 322, 'P-O': 335,
    'P-P': 201, 'PB-O': 382, 'PB-S': 346, 'RB-BR': 381, 'RB-CL': 428, 'RB-F': 494, 'RB-I': 319, 'S-H': 363,
    'S-O': 364, 'S-S': 268, 'SE-H': 276, 'SI-CL': 381, 'SI-F': 565, 'SI-H': 318, 'SI-O': 452, 'SI-SI': 222,
     'H-B': 389, 'H-GE': 288, 'H-SN': 251, 'H-AS': 247, 'H-TE': 238, 'B-B': 293, 'B-CL': 456, 'B-BR': 377,
      'C-GE': 238, 'C-SN': 192, 'C-PB': 130, 'C---O': 273, 'SI-N': 355, 'SI-S': 293, 'SI-BR': 310, 'SI-I': 234,
      'GE-GE': 188, 'GE-N': 257, 'GE-F': 470, 'GE-CL': 349, 'GE-BR': 276, 'GE-I': 212, 'SN-F': 414, 'SN-CL': 323,
       'SN-BR': 273, 'SN-I': 205, 'PB-F': 331, 'PB-CL': 243, 'PB-BR': 201, 'PB-I': 142, 'N-F': 283, 'N-CL': 313,
       'P--O': 209, 'P--S': 335, 'P-F': 490, 'P-I': 184, 'AS-AS': 146, 'AS-O': 301, 'AS-F': 484, 'AS-CL': 322,
       'AS-BR': 458, 'AS-I': 200, 'SB-SB': 121, 'SB-F': 440, 'O-F': 190, 'S--O': 522, 'S--S': 425, 'S-F': 284,
       'S-CL': 255, 'SE-SE': 172, 'SE--SE': 100, 'AT-AT': 116, 'I-O': 201, 'I-F': 273, 'I-CL': 208, 'I-BR': 175,
        'XE-O': 84, 'XE-F': 130, 'AROMATIC': 300}
BOND_ENERGY_BOUNDARIES = [-2, 0, 84.0, 188.0, 235.0, 272.0, 304.0, 336.0, 381.0, 456.0]

# (1) bond energy feature (2)bond length feature
def get_extra_bond_feature(bond: Chem.Bond):
    """Get bond features: bond energy, bond legth.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object

    Return:
    bond energy, bond length
    """
    bond_energy = get_bond_energy(bond)
    bond_energy_bucket = bucketize(bond_energy, BOND_ENERGY_BOUNDARIES)
    bond_energy_fea = np.array(bond_energy_bucket, dtype=np.float32)

    bond_len = get_bond_len(bond)
    bond_len_bucket = bucketize(bond_len, BOND_LEN_BOUNDARIES)
    bond_len_fea = np.array(bond_len_bucket, dtype=np.float32)

    return bond_energy_fea, bond_len_fea


# bond length
def get_bond_len(bond):
    a_1 = bond.GetBeginAtom().GetSymbol().upper()
    a_2 = bond.GetEndAtom().GetSymbol().upper()
    bt = bond.GetBondType()
    bt = BOND_TYPES.index(bt)
    if bt == 4: #AROMATIC
        return BOND_LEN_DICT.get('AROMATIC')
    elif a_1+'-'*bt+a_2 in BOND_LEN_DICT.keys():
        return BOND_LEN_DICT.get(a_1+'-'*bt+a_2)
    elif a_2+'-'*bt+a_1 in BOND_LEN_DICT.keys():
        return BOND_LEN_DICT.get(a_2+'-'*bt+a_1)
    else:
        return -1

# bond energy
def get_bond_energy(bond):
    a_1 = bond.GetBeginAtom().GetSymbol().upper()
    a_2 = bond.GetEndAtom().GetSymbol().upper()
    bt = bond.GetBondType()
    bt = BOND_TYPES.index(bt)
    if bt == 4: #AROMATIC
        return BOND_ENERGY_DICT.get('AROMATIC')
    elif a_1 + '-' * bt + a_2 in BOND_ENERGY_DICT.keys():
        return BOND_ENERGY_DICT.get(a_1 + '-' * bt + a_2)
    elif a_2 + '-' * bt + a_1 in BOND_ENERGY_DICT.keys():
        return BOND_ENERGY_DICT.get(a_2 + '-' * bt + a_1)
    else:
        return -1

#  feature bucket
def bucketize(value, boundaries):
    position = bisect.bisect_left(boundaries, value)
    res = [float(0)] * len(boundaries)
    res[position-1] = float(1)
    return res