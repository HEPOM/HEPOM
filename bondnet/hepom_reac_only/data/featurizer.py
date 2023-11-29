"""
Featurize a molecule homobidirected or complete graph with atom, bonds and global features
with RDKit
"""

from typing import Any
import torch
import os
import warnings
import itertools
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.rdchem import GetPeriodicTable
import networkx as nx
from hepom_reac_only.utils import *
from hepom_reac_only.data.utils import (
    one_hot_encoding_bn,
    rdkit_bond_desc,
    h_count_and_degree,
    get_dataset_species,
)

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

class BaseFeaturizer:
    def __init__(self, dtype='float32'):
        if dtype not in ['float32', 'float64']:
            raise ValueError(
                f"`dtype` should be `float32` or `float64`, but got {dtype}."
            )
        self.dtype = dtype
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        """
        Returns:
            The feature size in int format
        """
        return self._feature_size
    
    @property
    def feature_name(self):
        """
        Returns:
            a list of the names of each feature. Size should be same
            as `feature_size`.
        """

        return self._feature_name
    
    def __call__(self, mol, **kwargs):
        """
        Returns:
            A dictionary of the features
        """
        raise NotImplementedError
    

class BondAsEdgeBidirectedFeaturizer(BaseFeaturizer):
    """
    Featurize all bonds in the HomoBidirected molecule graph

    Feature of bond 0 is assigned to graph edges 0 and 1, feature of bond 1
    is assigned to graph edges 1 and 2. If `self_loop` is `True`,
    graph edge 2Nb, 2Nb + Na - 1 will also have features but they won't be
    determined from the bonds

    Args:
        self_loop(bool): Check for self loops in the graph
    """

    def __init__(
            self, 
            self_loop = True, 
            dtype='float32',
    ):
        self.self_loop = self_loop
        super().__init__()
    
    def __call__(self, mol, **kwargs):
        """
        Parameters
        --------------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        
        Returns
        ------------
            Dictionary for bond features
        """
        feats = []
        num_bonds = mol.GetNumBonds()

        if num_bonds < 1:
            warnings.warn('molecule graph has no edges/bonds')
        
        for u in range(num_bonds):
            bond = mol.GetBondWithIdx(u)

            ft = [
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
            ]

            ft += one_hot_encoding_bn(
                bond.GetBondType(),
                [
                 Chem.rdchem.BondType.SINGLE,
                 Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE,
                 None,
                ]
            )

            feats.extend([ft, ft])
        
        if self.self_loop:
            for i in range(mol.GetNumAtoms()):

                #use -1 to denote features when there is no bond as in self-loops

                ft = [-1, -1]

                #None type bond for self loops
                ft += one_hot_encoding_bn(
                    None,
                    [
                     Chem.rdchem.BondType.SINGLE,
                     Chem.rdchem.BondType.DOUBLE,
                     Chem.rdchem.BondType.TRIPLE,
                     None,
                    ]
                )

                feats.append(ft)
        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ['is in ring', 'is conjugated'] + ['bond type']*4

        return {'feat': feats}
    
class BondAsEdgeCompleteFeaturizer(BaseFeaturizer):
    """
    Featurize all bonds in the Complete molecule graph

    Features will be created between atoms pairs (0,0), (0,1)...
    and so on

    If not `self_loop`, (0,0), (1,1) will not be present

    Args:
        self_loop(bool): Check for self loops in the graph
    """

    def __init__(
            self, 
            self_loop = True, 
            dtype='float32',
    ):
        self.self_loop = self_loop
        super().__init__()
    
    def __call__(self, mol, **kwargs):
        """
        Parameters
        --------------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        
        Returns
        ------------
            Dictionary for bond features
        """
        feats = []
        num_bonds = mol.GetNumBonds()
        num_atoms = mol.GetNumAtoms()

        if num_bonds < 1:
            warnings.warn('molecule graph has no edges/bonds')
        
        for u in range(num_bonds):
            for v in range(num_atoms):
                if (u==v and not self.self_loop):
                    continue

            bond = mol.GetBondBetweenAtoms(u,v)
            if bond is None:
                bond_type = None
                ft = [-1, -1]
            
            else:
                bond_type = bond.GetBondType()
                ft = [
                    int(bond.IsInRing()),
                    int(bond.GetIsConjugated()),
                ]

                ft += one_hot_encoding_bn(
                bond.GetBondType(),
                [
                 Chem.rdchem.BondType.SINGLE,
                 Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE,
                 None,
                ]
                )
                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ['is in ring', 'is conjugated'] + ['bond type']*4

        return {'feat': feats}
    
class AtomFeaturizerMinimum(BaseFeaturizer):
    """
    Featurize the atoms in the molecule with minimum descriptors

    No hybridization info included
    """
    def __init__(
            self, 
            dtype='float32'
    ):
        super().__init__(dtype)

    def __call__(self, mol, **kwargs):
        """
        Parameters
        --------------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        `extra_feats_info`: should be provides as `kwargs` as additional info
    
        Returns
        ------------
        Dictionary for atom features
        """
        
        try:
            species = sorted(kwargs['dataset_species'])
        except KeyError as e:
            raise KeyError(
                f"{e} `dataset_species` needed for {self.__class__.__name__}"
            )
        
        try:
            feats_info = kwargs['extra_feats_info']
        except KeyError as e:
            raise KeyError(
                f"{e} `extra_feats_info` needed for {self.__class__.__name__}"
            )
        
        feats = []

        ring = mol.GetRingInfo()
        allowed_ring_size = [3, 4, 5, 6, 7, 8]
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            ft = []
            atom = mol.GetAtomWithIdx(i)

            ft.append(atom.GetTotalDegree())
            ft.append(int(atom.IsInRing()))
            ft.append(atom.GetTotalNumHs(includeNeighbors = True))

            ft += one_hot_encoding_bn(atom.GetSymbol(), species)

            for s in allowed_ring_size:
                ft.append(ring.IsAtomInRingOfSize(i, s))
            
            feats.append(ft)
        
        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ['total degree', 'is in ring', 'total H'] + 
            ['chemical symbol']*len(species) + 
            ['ring size']*6
        )

        return {'feat': feats}
        
class AtomFeaturizerFull(BaseFeaturizer):
        """
        Featurize the atoms in the molecule with an extended set of descriptors
        compared to AtomFeaturizerMinimum
        """
        def __init__(
                self, 
                dtype='float32'
        ):
            super().__init__(dtype)

        def __call__(self, mol, **kwargs):
            """
            Parameters
            --------------
            mol : rdkit.Chem.rdchem.Mol
                RDKit molecule object

            `extra_feats_info`: should be provides as `kwargs` as additional info
        
            Returns
            ------------
            Dictionary for atom features
            """
            
            try:
                species = sorted(kwargs['dataset_species'])
            except KeyError as e:
                raise KeyError(
                    f"{e} `dataset_species` needed for {self.__class__.__name__}"
                )
            
            feats = []

            ring = mol.GetRingInfo()
            allowed_ring_size = [3, 4, 5, 6, 7, 8]
            num_atoms = mol.GetNumAtoms()
            for i in range(num_atoms):
                ft = []
                atom = mol.GetAtomWithIdx(i)

                ft.append(atom.GetTotalDegree())
                ft.append(atom.GetTotalValence())
                ft.append(atom.GetNumRadicalElectrons())
                ft.append(int(atom.IsInRing()))
                ft.append(atom.GetTotalNumHs(includeNeighbors = True))

                ft += one_hot_encoding_bn(atom.GetSymbol(), species)

                ft += one_hot_encoding_bn(
                    atom.GetHybridization(),
                    [
                        Chem.rdchem.HybridizationType.S,
                        Chem.rdchem.HybridizationType.SP,
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                    ]
                )

                for s in allowed_ring_size:
                    ft.append(ring.IsAtomInRingOfSize(i, s))
                
                feats.append(ft)
            
            feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
            self._feature_size = feats.shape[1]
            self._feature_name = (
                [
                    'total degree', 
                    'total valence', 
                    'num radical electrons', 
                    'is in ring', 
                    'total H'] 
                + ['chemical symbol']*len(species) 
                + ['hybridization']*4
                + ['ring size']*6
            )

            return {'feat': feats}
        
class GlobalFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of the molecule graphs using
    number of atoms, number of bonds and molecular weight.

    Args:
        allowed_charges (list, optional): charges allowed for the molecules to take.
    """

    def __init__(self, allowed_charges = None, dtype='float32'):
        super().__init__(dtype)
        self.allowed_charges = allowed_charges

    def __call__(self, mol, **kwargs):

        pt = GetPeriodicTable()
        g = [
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            sum([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetNumAtoms()])
        ]

        if self.allowed_charges is not None:
            try:
                feats_info = kwargs['extra_feats_info']
            except KeyError as e:
                raise KeyError(
                    f"{e} `extra_feats_info` needed for {self.__class__.__name__}"
                )
            
            g += one_hot_encoding_bn(feats_info['charges'], self.allowed_charges)
        
        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            [
                'num atoms', 
                'num bonds', 
                'molecule weight', 
            ]
        )
        if self.allowed_charges is not None:
            self._feature_name += ['charge one hot']*len(self.allowed_charges)
        
        return {'feat': feats}

        
    
