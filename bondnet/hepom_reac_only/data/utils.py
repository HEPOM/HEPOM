import pandas as pd
import numpy as np
import networkx as nx
import itertools
import xyz2mol
import os
from rdkit import Chem
from collections import deque
from numpy import array
from sklearn.preprocessing import LabelEncoder

def get_dataset_species(molecules):
    """
    Get all the species of atoms in a set of RDKit molecules

    Args:
        molecules (list): rdkit molecules

    Returns:
        list: asequence of species string
    """
    system_species = set()
    for mol in molecules:
        if mol is None:
            continue
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        system_species.update(species)

    return sorted(system_species)

def one_hot_encoding_bn(x, allowable_set):
    """
    Returns the one-hot encoding of the species in every molecule

    Parameters
    -----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable set should be of the same type as x.

    Returns
    --------
    list
        List of int (0 or 1) where at most one value is 1.
        If the i-th value is 1 then x == allowable_set[i]
    """
    return list(map(int, list(map(lambda s: x==s, allowable_set))))

def h_count_and_degree(atom_ind, bond_list, species_order):
    """
    gets the number of H-atoms connected to an atom + degree of bonding
    input:
        atom_ind(int): index of atom
        bond_list(list of lists): list of bonds in graph
        species_order: order of atoms in graph to match nodes
    """
    atom_bonds = []
    h_count = 0
    for i in bond_list:
        if atom_ind in i:
            atom_bonds.append(i)
        
        if atom_bonds != 0:
            for bond in atom_bonds:
                bond_copy = bond[:]
                bond_copy.remove(atom_ind)
                if species_order[bond_copy[0]] == 'H':
                    h_count+=1
    return h_count, int(len(atom_bonds))

def rdkit_bond_desc(mol):
    """
    uses rdkit to get a dictionary with detected bond features but without any aromtiity
    information

    input:
        an rdkit molecule
    returns:
        ret_dict - a dictionary with the bonds(as root, target nodes) + descriptor info
    """

    detected_bonds_dict = {}
    allowed_bond_type = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ]

    num_atoms = len(mol.GetAtoms())

    for i in range(num_atoms):
        for j in range(num_atoms):
            try:
                bond = mol[0].GetBondBetweenAtoms(i, j)
            except:
                try:
                    bond = mol.GetBondBetweenAtoms(i, j)
                except:
                    ft = []
            
            if not bond is None:
                ft = one_hot_encoding_bn(bond.GetBondType(), allowed_bond_type)
                detected_bonds_dict[i, j] = ft

    return detected_bonds_dict

def get_rdkit_mols_from_path(directory):

    rdkit_mols = []

    try:
        os.listdir(directory)
    except NotADirectoryError as e:
        raise NotADirectoryError(
            f"{e} valid directory path needed for `get_rdkit_mols_from_path`"
            )
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        if os.path.isfile(f):

            try:
                xyz_file = xyz2mol.read_xyz_file(f)
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    f"{e} while processing file {f}"
                    )
                continue
            
            try:
                rdkit_mol = xyz2mol.xyz2mol(atoms=xyz_file[0], coordinates = xyz_file[2], charge = xyz_file[1])[0]
            except (IndexError, UnicodeDecodeError, TypeError, SystemExit) as error:
                print(f"{error} while processing file {f}")
                continue

            rdkit_mols.append(rdkit_mol)

    return(rdkit_mols)
        
def get_rdkit_mols_from_list(mol_list):
    
    rdkit_mols = []
    for smi in mol_list:
        try:
            r_mol = Chem.MolFromSmiles(smi)
            rdkit_mols.append(r_mol)
        except:
            continue
    
    rdkit_mols = [i for i in rdkit_mols if i is not None]
    return(rdkit_mols)

def get_aromatic_label(mol):
    lb = 0
    ar_at = 0
    try:
        num_atoms = mol.GetNumAtoms()
    except AttributeError as e:
        print(f"Supplied molecule is not a valid rdkit molecule")
    for at in range(num_atoms):
        if(mol.GetAtomWithIdx(at).GetIsAromatic()):
            ar_at += 1
    if (ar_at > 0):
        lb += 1
    
    return lb
