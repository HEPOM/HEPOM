from re import L
import time, copy
import pandas as pd
import networkx as nx
import numpy as np 
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from bondnet.core.reaction import Reaction
from bondnet.core.molwrapper import (
    rdkit_mol_to_wrapper_mol,
    create_wrapper_mol_from_atoms_and_bonds,
)
from bondnet.core.reaction import Reaction
from bondnet.core.reaction_collection import ReactionCollection
from bondnet.utils import int_atom, xyz2mol

from rdkit import Chem

Chem.WrapLogs()

def parse_extra_electronic_feats_atom(extra_feats, inds = None):

    valence_e = None 
    total_e = None 
    s = None 
    p = None
    d = None
    f = None
    occ = None
    spin = None 
    charges = None

    extra_feature_keys=list(extra_feats.keys())
    if("elec_occ" in extra_feature_keys):
        occ = extra_feats["elec_occ"]

    if("f_char" in extra_feature_keys):
        f = extra_feats["f_char"]

    if("d_char" in extra_feature_keys):
        d = extra_feats["d_char"]

    if("p_char" in extra_feature_keys):
        p = extra_feats["p_char"]

    if("s_char" in extra_feature_keys):
        s = extra_feats["s_char"]

    if("total_electrons" in extra_feature_keys):
        total_e = extra_feats["total_electrons"]

    if("valence_electrons" in extra_feature_keys):
        valence_e = extra_feats["valence_electrons"]

    #if("partial_spins" in extra_feature_keys):
    #    spin = extra_feats["partial_charges"]

    if("partial_charges" in extra_feature_keys):
        charges = extra_feats["partial_charges"]

    if(inds != None):
            valence_e = [valence_e[ind] for ind in inds]
            total_e = [total_e[ind] for ind in inds]
            charges = [charges[ind]for ind in inds]

            try: 
                s = [s[ind] for ind in inds]
                p = [p[ind] for ind in inds]
                d = [d[ind] for ind in inds]
                f = [f[ind] for ind in inds]
                occ = [occ[ind] for ind in inds]
            except: pass
                

    return valence_e, total_e, s, p, d, f, occ, spin, charges


def parse_extra_electronic_feats_bond(extra_feats, dict_bonds_as_root_target_inds):
    
    num_bonds = int(len(dict_bonds_as_root_target_inds.keys()))
    s_1 = [0 for i in range(num_bonds)]
    s_2 = [0 for i in range(num_bonds)]
    p_1 = [0 for i in range(num_bonds)]
    p_2 = [0 for i in range(num_bonds)]
    d_1 = [0 for i in range(num_bonds)]
    d_2 = [0 for i in range(num_bonds)]
    f_1 = [0 for i in range(num_bonds)]
    f_2 = [0 for i in range(num_bonds)]
    polar_1 = [0 for i in range(num_bonds)]
    polar_2 = [0 for i in range(num_bonds)]
    occ_nbo = [0 for i in range(num_bonds)]
    original_atom_ind = [0 for i in range(num_bonds)]
    
    extra_feature_keys=list(extra_feats.keys())
    if("indices_nbo" not in extra_feature_keys):
        return [], [], [], [], [], [], [], [], [], [], [], []
    extra_feat_bond_ind = extra_feats["indices_nbo"]
    extra_feat_bond_ind = [tuple(i) for i in extra_feat_bond_ind]

    if("1_s" in extra_feature_keys):
        s_1_temp = extra_feats["1_s"]
    if("2_s" in extra_feature_keys):
        s_2_temp = extra_feats["2_s"]
    if("1_p" in extra_feature_keys):
        p_1_temp = extra_feats["1_p"]
    if("2_p" in extra_feature_keys):
        p_2_temp = extra_feats["2_p"]
    if("1_d" in extra_feature_keys):
        d_1_temp = extra_feats["1_d"]
    if("2_d" in extra_feature_keys):
        d_2_temp = extra_feats["2_d"]
    if("1_f" in extra_feature_keys):
        f_1_temp = extra_feats["1_f"]
    if("2_f" in extra_feature_keys):
        f_2_temp = extra_feats["2_f"]
    if("1_polar" in extra_feature_keys):
        polar_1_temp = extra_feats["1_polar"]
    if("2_polar" in extra_feature_keys):
        polar_2_temp = extra_feats["2_polar"]
    if("occ_nbo" in extra_feature_keys):
        occ_nbo_temp = extra_feats["occ_nbo"]
 
    #for bond in bond_list:
    ind = 0
    for k, v in dict_bonds_as_root_target_inds.items(): 
        hit = 0 

        if(k in extra_feat_bond_ind):
            ind_in_extra = extra_feat_bond_ind.index(k)
            hit = True
        if((k[-1], k[0]) in extra_feat_bond_ind):
            ind_in_extra = extra_feat_bond_ind.index((k[-1], k[0]))
            hit = True
        if(hit): 
            s_1[ind] = s_1_temp[ind_in_extra]
            s_2[ind] = s_2_temp[ind_in_extra]
            p_1[ind] = p_1_temp[ind_in_extra]
            p_2[ind] = p_2_temp[ind_in_extra] 
            d_1[ind] = d_1_temp[ind_in_extra]
            d_2[ind] = d_2_temp[ind_in_extra]
            f_1[ind] = f_1_temp[ind_in_extra]
            f_2[ind] = f_2_temp[ind_in_extra]      
            polar_1[ind] = polar_1_temp[ind_in_extra]
            polar_2[ind] = polar_2_temp[ind_in_extra]      
            occ_nbo[ind] = occ_nbo_temp[ind_in_extra]
        ind+=1

    return s_1, s_2, p_1, p_2, d_1, d_2, f_1, f_2, polar_1, polar_2, occ_nbo, original_atom_ind


def split_and_map(
    species, bonds, coords, atom_count, reaction_scaffold, id, 
    bonds_nonmetal=None, charge=0, extra_feats_atom={}, extra_feats_bond={}):
    """
    takes a list of nodes+bonds+reaction bonds and computes subgraphs/species
    also returns mappings

    takes: 
        species(list of strs): list of elements 
        bonds(list of list/tuples): bond list 
        coords(list of list): atomic position
        atom_count(int): number of nodes/atoms in the total reaction
        reaction_scaffold(list of list/tuples): total bonds in rxn
        id(str): unique id 
        bonds_nonmetal(list of tuples/lists): list nonmetal bonds
        charge(int): charge for molecule
        extra_feats(dict): dictionary w/ extra features

    returns: 
        species(list of molwrappers)
        atom_map(dict): maps atomic positions to reaction scaffold 
        bond_mapp(dict): maps bonds in subgraphs to reaction template ind
    """    

    ret_list, bond_map = [], []
    id = str(id)
    
    G = nx.Graph()
    G.add_nodes_from([int(i) for i in range(atom_count)])
    for i in bonds: G.add_edge(i[0], i[1])  
    sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    
    if len(sub_graphs) >= 2:
        mapping = []
        for ind_sg, sg in enumerate(sub_graphs):
            dict_bonds, dict_bonds_as_root_target_inds = {}, {}
            bond_reindex_list, species_sg, coords_sg = [], [], [] 
            nodes = list(sg.nodes())
            # finds bonds mapped to subgraphs
            for origin_bond_ind in bonds:
                # check if root to edge is in node list for subgraph
                check = any(item in origin_bond_ind for item in nodes)
                if check: # if it is then map these to lowest values in nodes

                    bond_orig = nodes.index(origin_bond_ind[0]) 
                    bond_targ = nodes.index(origin_bond_ind[1])
                    bond_reindex_list.append([bond_orig, bond_targ])
                    # finds the index of these nodes in the reactant bonds
                    ordered_targ_obj = [np.min([origin_bond_ind[0], origin_bond_ind[1]]), 
                        np.max([origin_bond_ind[0], origin_bond_ind[1]])]

                    original_bond_index = reaction_scaffold.index(ordered_targ_obj)
                    dict_bonds_as_root_target_inds[tuple(ordered_targ_obj)] = (bond_orig, bond_targ)
                    dict_bonds[len(bond_reindex_list) - 1] = original_bond_index

            bond_map.append(dict_bonds)
            
            for site in nodes:
                species_sg.append(species[site])
                coords_sg.append(coords[site])
                
            mapping_temp = {i: ind for i, ind in enumerate(nodes)}
            mapping.append(mapping_temp)


            valence_e, total_e, s, p, d, f, occ, spin, charges = \
                parse_extra_electronic_feats_atom(extra_feats_atom, nodes)
            s_1, s_2, p_1, p_2, d_1, d_2, f_1, f_2, polar_1, polar_2, occ_nbo, _ = \
                parse_extra_electronic_feats_bond(extra_feats_bond, dict_bonds_as_root_target_inds)

            species_molwrapper = create_wrapper_mol_from_atoms_and_bonds(
                species_sg,
                coords_sg,
                bond_reindex_list,
                valence_e = valence_e, total_e = total_e, 
                s = s, p = p, d = d, f = f, 
                occ = occ, spin = spin, charges = charges, charge=charge,
                s_1=s_1,s_2=s_2, p_1=p_1, p_2=p_2, d_1=d_1, d_2=d_2, f_1=f_1,  
                f_2=f_2, polar_1=polar_1, polar_2=polar_2, occ_nbo=occ_nbo,
                identifier=id+"_" + str(ind_sg)
            )

            if(bonds_nonmetal==None): 
                species_molwrapper.nonmetal_bonds = bond_reindex_list
            else: 
                non_metal_filter = []
                for bond_non_metal in bonds_nonmetal:
                    if(bond_non_metal[0] in nodes or bond_non_metal[1] in nodes):
                        non_metal_filter.append(bond_non_metal)
                species_molwrapper.nonmetal_bonds = bonds_nonmetal
                
            ret_list.append(species_molwrapper)

    else:
        bond_reindex_list = []
        dict_temp, dict_bonds = {}, {}

        valence_e, total_e, s, p, d, f, occ, spin, charges = \
            parse_extra_electronic_feats_atom(extra_feats_atom)

        s_1, s_2, p_1, p_2, d_1, d_2, f_1, f_2, polar_1, polar_2, occ_nbo, _ = \
            parse_extra_electronic_feats_bond(extra_feats_bond, {tuple(i):tuple(i) for i in bonds})

        species_molwrapper = create_wrapper_mol_from_atoms_and_bonds(
            species,
            coords,
            bonds,
            valence_e = valence_e, total_e = total_e, 
            s = s, p = p, d = d, f = f, 
            occ = occ, spin = spin, charges = charges, charge=charge,
            s_1=s_1, s_2=s_2, p_1=p_1, p_2=p_2, d_1=d_1, d_2=d_2, f_1=f_1,  
            f_2=f_2, polar_1=polar_1, polar_2=polar_2, occ_nbo=occ_nbo,
            identifier=id
        )

        if(bonds_nonmetal==None): species_molwrapper.nonmetal_bonds = bonds
        else: species_molwrapper.nonmetal_bonds = bonds_nonmetal
        
        for origin_bond_ind in bonds:      
            nodes = list(G.nodes())
            check = any(item in origin_bond_ind for item in nodes)
            
            if check: # if it is then map these to lowest values in nodes
                bond_orig = nodes.index(origin_bond_ind[0]) 
                bond_targ = nodes.index(origin_bond_ind[1])
                bond_reindex_list.append([bond_orig, bond_targ])
                # finds the index of these nodes in the reactant bonds
                try:
                    original_bond_index = reaction_scaffold.index(
                        [np.min([origin_bond_ind[0], origin_bond_ind[1]]), 
                        np.max([origin_bond_ind[0], origin_bond_ind[1]])]
                    )

                    dict_bonds[len(bond_reindex_list) - 1] = original_bond_index
                except:  
                    print("detected bond in prod. not in reaction scaffold")
        bond_map = [dict_bonds]
        
        # atom map
        for i in range(len(species)):
            dict_temp[i] = i
        mapping = [dict_temp]
        ret_list.append(species_molwrapper)

    if(len(ret_list) != len(mapping)):print("ret list not equal to atom mapping list")
    if(len(ret_list) != len(bond_map)):print("ret list not equal to bond mapping list")
    
    return ret_list, mapping, bond_map


def process_species_graph(
    row, 
    classifier=False, 
    target='ts', 
    reverse_rxn=False, 
    verbose=False, 
    filter_species = None, 
    filter_outliers = False, 
    filter_sparse_rxns = False,
    lower_bound = -99, 
    upper_bound = 100,
    feature_filter = False, 
    categories = 5):
    """
    Takes a row and processes the products/reactants - entirely defined by graphs from row

    Args:
        row: the row (series) pandas object

    Returns:
        mol_list: a list of MolecularWrapper object(s) for product(s) or reactant
    """
    
    rxn = []

    if(filter_species == None): 
        filter_prod = -99
        filter_reactant = -99
    else: 
        filter_prod = filter_species[1]
        filter_reactant = filter_species[0]
        
    reactant_key = 'reactant'
    product_key = 'product'
    #reverse_rxn = False # generalize to augment with reverse
    
    charge = row["charge"]
    formed_len = len(row['bonds_formed'])
    broken_len = len(row['bonds_broken'])
    broken_bonds = [tuple(i) for i in row['bonds_broken']]
    formed_bonds = [tuple(i) for i in row['bonds_formed']]
    check_list_len = broken_len + formed_len
    
    if(check_list_len==0): 
        if(verbose):
            print("no bond changes detected")
        return 0

    else: 
        if(reverse_rxn):
            reactant_key = 'product'
            product_key = 'reactant'          
            formed_len = len(row['bonds_broken'])
            broken_len = len(row['bonds_formed'])
            formed_bonds = row['bonds_broken'] 
            broken_bonds = row['bonds_formed']

    if(broken_len == 0):
        temp_key = copy.deepcopy(reactant_key)
        reactant_key = product_key
        product_key = temp_key
        reverse_rxn = not reverse_rxn
        temp_broken = copy.deepcopy(broken_bonds)
        broken_bonds = formed_bonds
        formed_bonds = temp_broken
 
    bonds_reactant = row[reactant_key+"_bonds"]
    bonds_products = row[product_key+"_bonds"]
    try:
        pymat_graph_reactants = row["combined_" + reactant_key+"s_graph"]["molecule"]["sites"]
        pymat_graph_products = row["combined_" + reactant_key+"s_graph"]["molecule"]["sites"]
    except: 
        pymat_graph_reactants = row[reactant_key+"_molecule_graph"]["molecule"]["sites"]
        pymat_graph_products = row[product_key+"_molecule_graph"]["molecule"]["sites"]

    species_reactant = [int_atom(i["name"]) for i in pymat_graph_reactants]
    species_products_full = [int_atom(i["name"]) for i in pymat_graph_products]
    coords_reactant = [i["xyz"] for i in pymat_graph_reactants]
    coords_products_full = [i["xyz"] for i in pymat_graph_products]

    # new 
    total_bonds =  [tuple(bond) for bond in bonds_reactant]
    [total_bonds.append((np.min(np.array(i)), np.max(np.array(i)))) for i in bonds_products]

    total_bonds = list(set(total_bonds))
    total_bonds = [list(bond) for bond in total_bonds]
    
    num_nodes = 0
    for i in row["composition"].items():
        num_nodes += int(i[-1])

    bonds_nonmetal_product=row[product_key+'_bonds_nometal']
    bonds_nonmetal_reactant=row[reactant_key+'_bonds_nometal']
    
    # checks if there are other features to add to mol_wrapper object
    keys_list = list(row.index)

    extra_atom_feats_dict_prod, extra_atom_feats_dict_react = {}, {}
    extra_bond_feats_dict_prod, extra_bond_feats_dict_react = {}, {}
    
    if('reactant_partial_charges' in keys_list and 'product_partial_charges' in keys_list):
        extra_atom_feats_dict_prod["partial_charges"] = row[product_key+"_partial_charges"][0]
        extra_atom_feats_dict_react["partial_charges"] = row[reactant_key+"_partial_charges"][0]
    if('reactant_valence_electrons' in keys_list and 'product_valence_electrons' in keys_list):
        extra_atom_feats_dict_prod["valence_electrons"] = row[product_key+"_valence_electrons"][0]
        extra_atom_feats_dict_react["valence_electrons"] = row[reactant_key+"_valence_electrons"][0]
    if('reactant_total_electrons' in keys_list and 'product_total_electrons' in keys_list):
        extra_atom_feats_dict_prod["total_electrons"] = row[product_key+"_total_electrons"][0]
        extra_atom_feats_dict_react["total_electrons"] = row[reactant_key+"_total_electrons"][0]
    if('reactant_s_char' in keys_list and 'product_s_char' in keys_list):
        extra_atom_feats_dict_prod["s_char"] = row[product_key+"_s_char"][0]
        extra_atom_feats_dict_react["s_char"] = row[reactant_key+"_s_char"][0]
    if('reactant_p_char' in keys_list and 'product_p_char' in keys_list):
        extra_atom_feats_dict_prod["p_char"] = row[product_key+"_p_char"][0]
        extra_atom_feats_dict_react["p_char"] = row[reactant_key+"_p_char"][0]
    if('reactant_d_char' in keys_list and 'product_d_char' in keys_list):
        extra_atom_feats_dict_prod["d_char"] = row[product_key+"_d_char"][0]
        extra_atom_feats_dict_react["d_char"] = row[reactant_key+"_d_char"][0]
    if('reactant_f_char' in keys_list and 'product_f_char' in keys_list):
        extra_atom_feats_dict_prod["f_char"] = row[product_key+"_f_char"][0]
        extra_atom_feats_dict_react["f_char"] = row[reactant_key+"_f_char"][0]
    if('reactant_elec_occ' in keys_list and 'product_elec_occ' in keys_list):
        extra_atom_feats_dict_prod["elec_occ"] = row[product_key+"_elec_occ"][0]
        extra_atom_feats_dict_react["elec_occ"] = row[reactant_key+"_elec_occ"][0]

    if('product_1_s' in keys_list and 'reactant_1_s' in keys_list):
        extra_bond_feats_dict_prod["1_s"] = row[product_key+"_1_s"][0]
        extra_bond_feats_dict_react["1_s"] = row[reactant_key+"_1_s"][0]
    if('product_2_s' in keys_list and 'reactant_2_s' in keys_list):
        extra_bond_feats_dict_prod["2_s"] = row[product_key+"_2_s"][0]
        extra_bond_feats_dict_react["2_s"] = row[reactant_key+"_2_s"][0]
    if('product_1_p' in keys_list and 'reactant_1_p' in keys_list):
        extra_bond_feats_dict_prod["1_p"] = row[product_key+"_1_p"][0]
        extra_bond_feats_dict_react["1_p"] = row[reactant_key+"_1_p"][0]
    if('product_2_p' in keys_list and 'reactant_2_p' in keys_list):
        extra_bond_feats_dict_prod["2_p"] = row[product_key+"_2_p"][0]
        extra_bond_feats_dict_react["2_p"] = row[reactant_key+"_2_p"][0]
    if('product_1_d' in keys_list and 'reactant_1_d' in keys_list):
        extra_bond_feats_dict_prod["1_d"] = row[product_key+"_1_d"][0]
        extra_bond_feats_dict_react["1_d"] = row[reactant_key+"_1_d"][0]
    if('product_2_d' in keys_list and 'reactant_2_d' in keys_list):
        extra_bond_feats_dict_prod["2_d"] = row[product_key+"_2_d"][0]
        extra_bond_feats_dict_react["2_d"] = row[reactant_key+"_2_d"][0]
    if('product_1_f' in keys_list and 'reactant_1_f' in keys_list):
        extra_bond_feats_dict_prod["1_f"] = row[product_key+"_1_f"][0]
        extra_bond_feats_dict_react["1_f"] = row[reactant_key+"_1_f"][0]
    if('product_2_f' in keys_list and 'reactant_2_f' in keys_list):
        extra_bond_feats_dict_prod["2_f"] = row[product_key+"_2_f"][0]
        extra_bond_feats_dict_react["2_f"] = row[reactant_key+"_2_f"][0]

    if('reactant_1_polar' in keys_list and 'product_2_polar' in keys_list):
        extra_bond_feats_dict_prod["1_polar"] = row[product_key+"_1_polar"][0]
        extra_bond_feats_dict_react["1_polar"] = row[reactant_key+"_1_polar"][0]
    if('reactant_2_polar' in keys_list and 'product_2_polar' in keys_list):
        extra_bond_feats_dict_prod["2_polar"] = row[product_key+"_2_polar"][0]
        extra_bond_feats_dict_react["2_polar"] = row[reactant_key+"_2_polar"][0]
    if('reactant_occ_nbo' in keys_list and 'product_occ_nbo' in keys_list):
        extra_bond_feats_dict_prod["occ_nbo"] = row[product_key+"_occ_nbo"][0]
        extra_bond_feats_dict_react["occ_nbo"] = row[reactant_key+"_occ_nbo"][0]

    if('reactant_indices_nbo' in keys_list and 'product_indices_nbo' in keys_list):
        extra_bond_feats_dict_prod["indices_nbo"] = row[product_key+"_indices_nbo"][0]
        extra_bond_feats_dict_react["indices_nbo"] = row[reactant_key+"_indices_nbo"][0]


    if(feature_filter): 
        for k, v in extra_bond_feats_dict_prod.items(): 
            if(v == []): return []
        for k, v in extra_bond_feats_dict_react.items(): 
            if(v == []): return []
        for k, v in extra_atom_feats_dict_prod.items(): 
            if(v == []): return []
        for k, v in extra_atom_feats_dict_react.items(): 
            if(v == []): return []

    products, atoms_products, mapping_products = split_and_map(
        species=species_products_full,
        coords=coords_products_full,
        bonds=row[product_key+"_bonds"], 
        atom_count=num_nodes, 
        reaction_scaffold=total_bonds,
        id=str(row[product_key+"_id"]), 
        bonds_nonmetal=bonds_nonmetal_product,
        charge=charge,
        extra_feats_atom = extra_atom_feats_dict_prod,
        extra_feats_bond = extra_bond_feats_dict_prod)
    
    reactants, atoms_reactants, mapping_reactants = split_and_map(
        species=species_reactant,
        coords=coords_reactant,
        bonds = row[reactant_key+"_bonds"], 
        atom_count = num_nodes, 
        reaction_scaffold=total_bonds,
        id = str(row[reactant_key+"_id"]), 
        bonds_nonmetal=bonds_nonmetal_reactant,
        charge=charge, 
        extra_feats_atom = extra_atom_feats_dict_react,
        extra_feats_bond = extra_bond_feats_dict_react)
    
    total_atoms=list(set(list(np.concatenate([list(i.values()) for i in atoms_reactants]).flat)))
    check=False
    if(check):
        total_atoms_check=list(set(list(np.concatenate([list(i.values()) for i in atoms_products]).flat)))
        assert (total_atoms==total_atoms_check), 'atoms in reactant and products are not equal'
    
    if products != [] and reactants != []:
        rxn_type = [] 

        if(filter_prod != -99 or filter_reactant != -99):
            if(len(products) > filter_prod):
                return rxn  
        if(filter_reactant != -99):
            if(len(reactants) > filter_reactant):
                return rxn  

        try:
            id = [i for i in row["reaction_id"].split("-")]
            id = int(id[0] + id[1] + id[2])
        except: 
            id = row["reactant_id"]
            if(type(row["product_id"]) == list): 
                for i in row["product_id"]:
                    id+=i
            else: id+= row["product_id"]
            id = int(id)


        if(target == 'ts'):
            value = row['transition_state_energy'] - row[reactant_key+'_energy']
            reverse_energy = row['transition_state_energy'] - row[product_key+'_energy']
            if(reverse_energy < 0.0): reverse_energy = 0.0
            if(value < 0.0): value = 0.0
        elif(target == 'dG_sp'):
            value = row['dG_sp']
            reverse_energy = -value
        else:
            value = row[product_key+'_energy'] - row[reactant_key+'_energy']
            reverse_energy = row[reactant_key+'_energy'] - row[product_key+'_energy']
        
        if classifier:
            if(categories == 3):
                if value <= 0.1:
                    value = 0
                elif value < 0.7 and value > 0.1:
                    value = 1
                else:
                    value = 2

                if reverse_energy <= 0.1:
                    reverse_energy = 0
                elif reverse_energy < 0.7 and reverse_energy > 0.1:
                    reverse_energy = 1
                else:
                    reverse_energy = 2

            else: 
                if value <= 0.04:
                    value = 0
                elif value < 0.3 and value > 0.04:
                    value = 1
                elif value < 0.7 and value > 0.3:
                    value = 2
                elif value < 1.5 and value > 0.7:
                    value = 3
                else:
                    value = 4

                if reverse_energy <= 0.04:
                    reverse_energy = 0
                elif reverse_energy < 0.3 and reverse_energy > 0.04:
                    reverse_energy = 1
                elif reverse_energy < 0.7 and reverse_energy > 0.3:
                    reverse_energy = 2
                elif reverse_energy < 1.5 and reverse_energy > 0.7:
                    reverse_energy = 3
                else:
                    reverse_energy = 4
                
        if(len(broken_bonds) > 0 ):
            for i in broken_bonds:
                key = 'broken_'
                index = i 

                try:
                    atom_1 = row["combined_" + reactant_key+"s_graph"]["molecule"]["sites"][index[0]]["name"]
                    atom_2 = row["combined_" + reactant_key+"s_graph"]["molecule"]["sites"][index[1]]["name"] 
                except: 
                    atom_1 = row[reactant_key+"_molecule_graph"]["molecule"]["sites"][index[0]]["name"]
                    atom_2 = row[reactant_key+"_molecule_graph"]["molecule"]["sites"][index[1]]["name"]
                
                atoms = [atom_1, atom_2]
                atoms.sort()
                key += atoms[0] + "_" + atoms[1]
                rxn_type.append(key)

        if(len(formed_bonds) > 0):
            for i in formed_bonds:
                key = 'formed_'
                index = i 
                try:
                    atom_1 = row["combined_" + reactant_key+"s_graph"]["molecule"]["sites"][index[0]]["name"]
                    atom_2 = row["combined_" + reactant_key+"s_graph"]["molecule"]["sites"][index[1]]["name"] 
                except: 
                    atom_1 = row[reactant_key+"_molecule_graph"]["molecule"]["sites"][index[0]]["name"]
                    atom_2 = row[reactant_key+"_molecule_graph"]["molecule"]["sites"][index[1]]["name"]
             
                atoms = [atom_1, atom_2]
                atoms.sort()
                key += atoms[0] + "_" + atoms[1]
                rxn_type.append(key)
        
        if(filter_sparse_rxns):
            filter_rxn_list = [
                'broken_C_C', 'broken_C_Cl', 'broken_C_F', 'broken_C_H',
            'broken_C_Li', 'broken_C_N',
            'broken_C_O', 'broken_H_Li',
            'broken_F_H', 'broken_H_N',
            'broken_H_O', 'broken_Li_O',
            'formed_C_C', 'formed_C_Cl',
            'formed_C_F', 'formed_C_H',
            'formed_C_Li', 'formed_C_N',
            'formed_C_O', 'formed_H_Li',
            'formed_F_H',  'formed_H_H',
            'formed_H_O', 'formed_Li_O']

            check = any(item in rxn_type for item in filter_rxn_list)
            if (check == False): 
                print("filtering rxn")
                return []

        rxn = Reaction(
            reactants=reactants,
            products=products,
            free_energy=value,
            broken_bond=broken_bonds,
            formed_bond=formed_bonds,
            total_bonds=total_bonds,
            total_atoms=total_atoms,
            reverse_energy_target=reverse_energy,
            identifier=id,
            reaction_type = rxn_type 
        )
        atom_mapping_check = []
        for i in atoms_reactants: 
            for key in i.keys(): 
                atom_mapping_check.append(i[key])
        atom_mapping_check = list(set(atom_mapping_check))
        if(atom_mapping_check!=total_atoms): print(atom_mapping_check, total_atoms)
        
        rxn.set_atom_mapping([atoms_reactants, atoms_products])
        rxn._bond_mapping_by_int_index = [mapping_reactants, mapping_products]
        
        outlier_condition = lower_bound > value or upper_bound < value
        if(outlier_condition and filter_outliers): return []
    
    return rxn


def process_species_rdkit(row, classifier=False):
    """
    Takes a row and processes the products/reactants - entirely defined by rdkit definitions

    Args:
        row: the row (series) pandas object

    Returns:
        mol_list: a list of MolecularWrapper object(s) for product(s) or reactant
    """
    fail = 0
    rxn, reactant_list, product_list, bond_map = [], [], [], []
    reactant_key = 'reactant'
    product_key = 'product'
    
    reverse_rxn = False
    if(row['bonds_broken'] == [] and row['bonds_formed'] != []):
        reverse_rxn = True
        reactant_key = 'product'
        product_key = 'reactant'
    
    if(row['bonds_broken'] == [] and row['bonds_formed'] == []):
        return rxn

    species_reactant = [
        int_atom(i["name"]) for i in row[reactant_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    species_products_full = [
        int_atom(i["name"]) for i in row[product_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    coords_reactant = [
        i["xyz"] for i in row[reactant_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    coords_products_full = [
        i["xyz"] for i in row[product_key+"_molecule_graph"]["molecule"]["sites"]
    ]

    charge = row["charge"]
    id = str(row[reactant_key+"_id"])
    free_energy = row[product_key+"_free_energy"]

    reactant_mol = xyz2mol(
        atoms=species_reactant,
        coordinates=coords_reactant,
        charge=charge,
    )
    reactant_wrapper = rdkit_mol_to_wrapper_mol(
        reactant_mol[0], charge=charge, free_energy=free_energy, identifier=id
    )
    reactant_list.append(reactant_wrapper)

    # handle products
    # check subgraphs first
    num_nodes = 0
    for i in row["composition"].items():
        num_nodes += int(i[-1])
    G = nx.Graph()
    G.add_nodes_from([int(i) for i in range(num_nodes)])
    for i in row[product_key+"_bonds"]:
        G.add_edge(i[0], i[1])  # rdkit bonds are a subset of user-defined bonds
    sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    id = str(row[product_key+"_id"])

    # still no handling for rxns A --> B + C +....
    if len(sub_graphs) > 2:
        pass #print("cannot handle three or more products")
    # handle A --> B + C
    elif len(sub_graphs) == 2:
        mapping, mol_prod = [], []
        for sg in sub_graphs:

            coords_products, species_products, bond_reindex_list = [], [], []
            nodes = list(sg.nodes())
            bonds = list(sg.edges())

            # finds bonds mapped to subgraphs
            for origin_bond_ind in row[product_key+"_bonds"]:
                # check if root to edge is in node list for subgraph
                check = any(item in origin_bond_ind for item in nodes)
                if check:
                    bond_orig = nodes.index(origin_bond_ind[0])
                    bond_targ = nodes.index(origin_bond_ind[1])
                    bond_reindex_list.append([bond_orig, bond_targ])

            for site in nodes:
                species_products.append(
                    int_atom(
                        row[product_key+"_molecule_graph"]["molecule"]["sites"][site]["name"]
                    )
                )
                coords_products.append(
                    row[product_key+"_molecule_graph"]["molecule"]["sites"][site]["xyz"]
                )

            mapping_temp = {i: ind for i, ind in enumerate(nodes)}
            mapping.append(mapping_temp)

            mol = xyz2mol(
                atoms=species_products,
                coordinates=coords_products,
                charge=charge,
            )[0]
            product = rdkit_mol_to_wrapper_mol(
                mol, charge=charge, free_energy=free_energy, identifier=id
            )
            product_list.append(product)
    else:
        mol_prod = xyz2mol(
            atoms=species_products_full,
            coordinates=coords_products_full,
            charge=charge,
        )[0]
        product = rdkit_mol_to_wrapper_mol(
            mol_prod, charge=charge, free_energy=free_energy, identifier=id
        )
        product_list.append(product)

        # atom mapping - order is preserved
        dict_temp = {}
        for i in range(len(species_products_full)):
            dict_temp[i] = i
        mapping = [dict_temp]

    if fail == 0 and product_list != [] and reactant_list != []:
        id = [i for i in row["reaction_id"].split("-")]
        id = int(id[0] + id[1] + id[2])
        broken_bond = None

        if row["bonds_broken"] != []:
            broken_bond = row["bonds_broken"][0]
        if(reverse_rxn):
            broken_bond = row["bonds_formed"][0]

        if(reverse_rxn):
            value = row['transition_state_energy'] - row['product_energy']
        else: 
            value = row["dE_barrier"]

        if classifier:
            if value <= 0.04:
                value = 0
            elif value < 0.3 and value > 0.04:
                value = 1
            elif value < 0.7 and value > 0.3:
                value = 2
            elif value < 1.5 and value > 0.7:
                value = 3
            else:
                value = 4

        rxn = Reaction(
            reactants=reactant_list,
            products=product_list,
            free_energy=value,
            broken_bond=broken_bond,
            identifier=id,
        )
        rxn.set_atom_mapping(mapping)
    return rxn


def task_done(future):
    try:
        result = future.result()  # blocks until results are ready
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback)  # traceback of the function


def create_struct_label_dataset_bond_based_regression(filename, out_file):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for training
    """
    path_mg_data = "../../home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()

    # pebble implementation
    rxn_raw = []
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        for _, row in mg_df.head(50).iterrows():
            future = pool.schedule(process_species_graph, args=[row], timeout=30)
            rxn_raw.append(future.result())
    print(rxn_raw)
    finish_time = time.perf_counter()
    # pool.close()
    # pool.join()

    print(f"Program finished in {finish_time-start_time} seconds")

    for rxn_temp in rxn_raw:
        if not isinstance(rxn_temp, list):
            reactions.append(rxn_temp)

    print("number of rxns counted: " + str(len(reactions)))
    extractor = ReactionCollection(reactions)

    # works
    extractor.create_struct_label_dataset_bond_based_regression(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
        group_mode="charge_0",
    )


def create_struct_label_dataset_reaction_network(filename, out_file):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    """
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()
    # pebble implementation
    rxn_raw = []
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        for _, row in mg_df.head(50).iterrows():
            process_species = process_species_rdkit
            future = pool.schedule(process_species, args=[row], timeout=30)
            future.add_done_callback(task_done)
            try:
                rxn_raw.append(future.result())
            except:
                pass
    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
    for rxn_temp in rxn_raw:
        if not isinstance(rxn_temp, list):
            reactions.append(rxn_temp)

    print("number of rxns counted: " + str(len(reactions)))
    # molecules = get_molecules_from_reactions(reactions)
    extractor = ReactionCollection(reactions)

    # works
    extractor.create_struct_label_dataset_reaction_based_regression(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
        group_mode="charge_0",
    )


def create_reaction_network_files(filename, out_file, classifier=False):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    """
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()

    rxn_raw = []
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        for _, row in mg_df.head(100).iterrows():
            # process_species = process_species_rdkit
            future = pool.schedule(process_species_rdkit, args=[row, True], timeout=30)
            future.add_done_callback(task_done)
            try:
                rxn_raw.append(future.result())
            except:
                pass
    finish_time = time.perf_counter()
    print("rxn raw len: {}".format(int(len(rxn_raw))))
    print(f"Program finished in {finish_time-start_time} seconds")
    fail_default, fail_count, fail_sdf_map, fail_prod_len = 0, 0, 0, 0
    for rxn_temp in rxn_raw:
        if not isinstance(rxn_temp, list):  # bunch of stuff is being filtered here
            try:
                rxn_temp.get_broken_bond()
                try:
                    bond_map = rxn_temp.bond_mapping_by_sdf_int_index()
                    rxn_temp._bond_mapping_by_int_index = bond_map

                    reactant_bond_count = int(
                        len(rxn_temp.reactants[0].rdkit_mol.GetBonds())
                    )
                    prod_bond_count = 0
                    for i in rxn_temp.products:
                        prod_bond_count += int(len(i.rdkit_mol.GetBonds()))
                    if reactant_bond_count < prod_bond_count:
                        fail_prod_len += 1
                    else:
                        reactions.append(rxn_temp)
                except:
                    fail_sdf_map += 1
            except:
                fail_count += 1
        else:
            fail_default += 1

    print(".............failures.............")
    print("reactions len: {}".format(int(len(reactions))))
    print("bond break fail count: \t\t{}".format(fail_count))
    print("default fail count: \t\t{}".format(fail_default))
    print("sdf map fail count: \t\t{}".format(fail_sdf_map))
    print("product bond fail count: \t{}".format(fail_prod_len))

    extractor = ReactionCollection(reactions)
    # works
    if classifier:
        (
            all_mols,
            all_labels,
            features,
        ) = extractor.create_struct_label_dataset_reaction_based_regression_alt(
            struct_file=path_mg_data + "mg_struct_bond_rgrn_classify.sdf",
            label_file=path_mg_data + "mg_label_bond_rgrn_classify.yaml",
            feature_file=path_mg_data + "mg_feature_bond_rgrn_classify.yaml",
            group_mode="charge_0",
        )

    else:
        (
            all_mols,
            all_labels,
            features,
        ) = extractor.create_struct_label_dataset_reaction_based_regression_alt(
            struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
            label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
            feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
            group_mode="charge_0",
        )
    
    return all_mols, all_labels, features


def create_reaction_network_files_and_valid_rows(filename, 
    out_file, 
    bond_map_filter=False, 
    target='ts', 
    classifier=False, 
    debug=False, 
    augment=False, 
    filter_species = False, 
    filter_outliers = True,
    filter_sparse_rxn = False,
    feature_filter = False, 
    categories = 5):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
        bond_map_filter: true uses filter with sdf
        target (str): target for regression either 'ts' or 'diff'
        classifier(bool): whether to create a classification or reg. task 
        debug(bool): use smaller dataset or not
    """

    #path_mg_data = "../../../dataset/mg_dataset/20220613_reaction_data.json"
    path_json = filename
    
    print("reading file from: {}".format(path_json))
    mg_df = pd.read_json(path_json) 

    start_time = time.perf_counter()
    reactions, ind_val, rxn_raw, ind_final = [], [], [], []
    lower_bound, upper_bound = 0, 0

    if(debug): mg_df = mg_df.head(100)

    if(filter_outliers):
        de_barr = mg_df["dE_barrier"] 
        q1, q3, med = np.quantile(de_barr, 0.25), np.quantile(de_barr, 0.75), np.median(de_barr)
        # finding the iqr region
        iqr = q3-q1
        # finding upper and lower whiskers
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)

    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        
        for ind, row in mg_df.iterrows(): 
            future = pool.schedule(process_species_graph, 
                        args=[row], 
                        kwargs={"classifier":classifier,
                                "target":target,
                                "reverse_rxn":False,
                                "verbose": False,
                                "categories": categories,
                                "filter_species": filter_species,
                                "filter_outliers":filter_outliers,
                                "upper_bound":upper_bound,
                                "lower_bound":lower_bound,
                                "filter_sparse_rxns":filter_sparse_rxn,
                                "feature_filter": feature_filter},
                        timeout=30)
            future.add_done_callback(task_done)
            try:
                rxn_raw.append(future.result())
                ind_val.append(ind)
            except:
                pass
            if(augment):
                future = pool.schedule(process_species_graph, args=[row, classifier, target, True], timeout=30)
                future.add_done_callback(task_done)
                try:
                    rxn_raw.append(future.result())
                    ind_val.append(ind)
                except:
                    pass               
    finish_time = time.perf_counter()
    print("rxn raw len: {}".format(int(len(rxn_raw))))
    print(f"Program finished in {finish_time-start_time} seconds")
    fail_default, fail_count, fail_sdf_map, fail_prod_len = 0, 0, 0, 0
    
    for ind, rxn_temp in enumerate(rxn_raw):
        if not isinstance(rxn_temp, list):  
            try:
                rxn_temp.get_broken_bond()
                if(bond_map_filter):
                    try:
                        bond_map = rxn_temp.bond_mapping_by_int_index()
                        rxn_temp._bond_mapping_by_int_index = bond_map
                        reactant_bond_count = int(
                            len(rxn_temp.reactants[0].rdkit_mol.GetBonds())
                        ) # here we're using rdkit still
                        prod_bond_count = 0
                        for i in rxn_temp.products:
                            prod_bond_count += int(len(i.rdkit_mol.GetBonds()))
                        if reactant_bond_count < prod_bond_count:
                            fail_prod_len += 1
                        else:
                            reactions.append(rxn_temp)
                            ind_final.append(ind_val[ind])
                    except:
                        fail_sdf_map += 1
                else:         
                    bond_map = rxn_temp.bond_mapping_by_int_index()
                    
                    if(len(rxn_temp.reactants) != len(bond_map[0]) or
                    len(rxn_temp.products) != len(bond_map[1])):
                        print("mappings invalid")
                        fail_count += 1
                    else:                        
                        rxn_temp._bond_mapping_by_int_index = bond_map
                        reactions.append(rxn_temp)
                        ind_final.append(ind_val[ind])
            except:
                fail_count += 1
        else:
            fail_default += 1

    print(".............failures.............")
    print("reactions len: {}".format(int(len(reactions))))
    print("valid ind len: {}".format(int(len(ind_final))))
    print("bond break fail count: \t\t{}".format(fail_count))
    print("default fail count: \t\t{}".format(fail_default))
    print("sdf map fail count: \t\t{}".format(fail_sdf_map))
    print("product bond fail count: \t{}".format(fail_prod_len))
    print("about to group and organize")
    

    extractor = ReactionCollection(reactions)
    (
        all_mols,
        all_labels,
        features,
    
    ) = extractor.create_struct_label_dataset_reaction_based_regression_general(
        struct_file=path_json + "mg_struct_bond_rgrn_classify.sdf",
        label_file=path_json + "mg_label_bond_rgrn_classify.yaml",
        feature_file=path_json + "mg_feature_bond_rgrn_classify.yaml",
        group_mode="charge_0",
        sdf_mapping=False,
    )

    return all_mols, all_labels, features


def process_data():
    # create_struct_label_dataset_reaction_network(filename='', out_file='./')
    # create_struct_label_dataset_bond_based_regression(filename='', out_file='./')
    all_mols, all_labels, features = create_reaction_network_files(
        filename="", out_file="./", target = 'ts'
    )
    return all_mols, all_labels, features


#if __name__ == "__main__":
#    process_data()
