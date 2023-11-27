"""
Build molecule graph using PyG and then featurize it.
"""
import itertools
import numpy as np

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.utils.data import DataLoader

class BaseGraph:
    """
    Base grapher to build PyG graphs and featurize them. Typically should not
    use this directly.
    """

    def __init__(
            self,
            atom_featurizer=None,
            bond_featurizer=None,
            self_loop=False,
    ):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.self_loop = self_loop

    def build_graph(self, mol):
        raise NotImplementedError
    
    def featurize(self, g, mol, **kwargs):
        raise NotImplementedError
    
    def build_graph_and_featurize(self, mol, **kwargs):
        """
        Build a graphs with atoms as the nodes and bonds as the edges of this 
        graph

        Args:
            mol(rdkit mol): a rdkit molecule
            kwargs: any extra keyword arguments needed by the featurizer

        Returns:
            PyG graph
        """

        g = self.build_graph(mol)
        g = self.featurize(g, mol, **kwargs)
        return g

    @property
    def feature_size(self):
        res = {}
        if self.atom_featurizer is not None:
            res['atom'] = self.atom_featurizer.feature_size
        if self.bond_featurizer is not None:
            res['bond'] = self.bond_featurizer.feature_size
        if hasattr(self, "global_featurizer") and self.global_featurizer is not None:
            res['global'] = self.global_featurizer.feature_size
        return res
    
    @property
    def feature_name(self):
        res = {}
        if self.atom_featurizer is not None:
            res['atom'] = self.atom_featurizer.feature_name
        if self.bond_featurizer is not None:
            res['bond'] = self.bond_featurizer.feature_name
        if hasattr(self, "global_featurizer") and self.global_featurizer is not None:
            res['global'] = self.global_featurizer.feature_name
        return res
    
class HomoBidirectedGraph(BaseGraph):
    """
    Convert a RDKit molecule to a homogenous bidirected PyG graph and then featurize it.

    Class creates a bidirectional graph in which Atom i corresponds to node i of the graph.
    Bond 0 corresponds to edge(bond) between node 0 and 1, Bond 1 corresponds to graph edge
    between node 1 and 2... and so on. If 'self_loop' is True then edge 2N will correspond
    to the self loop of atom 0, 2N+1 will be the self loop of atom 1... N being the number
    of edges(bonds) in the molecule graph

    Notes:
        Make sure the featurizer matches the above order, especially bond_featurizer
    """

    def __init__(self, atom_featurizer = None, bond_featurizer = None, self_loop = True):
        super(HomoBidirectedGraph, self).__init__(
            atom_featurizer, bond_featurizer, self_loop
        )

    def build_graph(self, mol):
        
        num_atoms = mol.GetNumAtoms()
        
        atom_feature_size = 1 #Arbitrary, will be reassigned by featurizer
        node_data = np.zeros((num_atoms, atom_feature_size))
        node_data = torch.tensor(node_data, dtype=torch.float)

        #construct edge_index array E of shape (2, 2*n_bonds)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        
        E = torch.stack((torch_rows, torch_cols), dim=0)
        if (self.self_loop):
            E = add_self_loops(edge_index=E, num_nodes=num_atoms)[0] #added self_loops

        g = Data(x=node_data, edge_index=E)

        #add name to molecule graph
        g.mol_name = Chem.MolToSmiles(mol, kekuleSmiles=True)

        return g
    
    def featurize(self, g, mol, **kwargs):
        if self.atom_featurizer is not None:
            g.x = self.atom_featurizer(mol, **kwargs)['feat']
        if self.bond_featurizer is not None:
            g.edge_attr = self.bond_featurizer(mol, **kwargs)['feat']
        #if self.global_featurizer is not None:
        #   g.u = self.global_featurizer(mol, **kwargs)
        return g
    
class HomoCompleteGraph(BaseGraph):
    """
    Convert a RDKit molecule to a homogenous bidirected PyG graph and then featurize it.

    Class creates a complete graph, i.e. every atom is connected to other atoms in the
    molecule. If 'self_loop' is 'True' atom is also connected to itself
    """

    def __init__(self, atom_featurizer=None, bond_featurizer=None, self_loop=False):
        super(HomoCompleteGraph).__init__(
            atom_featurizer, bond_featurizer, self_loop
            )
    
    def build_graph(self, mol):
        num_atoms = mol.GetNumAtoms()

        atom_feature_size = self.feature_size['atom'] #Check dimensionality

        node_data = np.zeros((num_atoms, atom_feature_size))
        node_data = torch.tensor(node_data, dtype=torch.float)

        #construct edge_index array E of shape (2, 2*n_bonds)
        adj_mat_com = np.ones(GetAdjacencyMatrix(mol).shape, int)
        if (not self.self_loop):
            np.fill_diagonal(adj_mat_com, 0)
        
        (rows, cols) = np.nonzero(adj_mat_com)
        
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        
        E = torch.stack((torch_rows, torch_cols), dim=0)

        g = Data(x=node_data, edge_index=E)

        #add name to molecule graph
        g.mol_name = Chem.MolToSmiles(mol, kekuleSmiles=True)

        return g
    
    def featurize(self, g, mol, **kwargs):
        if self.atom_featurizer is not None:
            g.x = self.atom_featurizer(mol, **kwargs)['feat']
        if self.bond_featurizer is not None:
            g.edge_attr = self.bond_featurizer(mol, **kwargs)['feat']
        # if self.global_featurizer is not None:
        #     g.u = self.global_featurizer(mol, **kwargs)
        return g
        
            
    




    
