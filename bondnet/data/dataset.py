import torch, itertools
import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import defaultdict, OrderedDict
from concurrent.futures import TimeoutError
from tqdm import tqdm
from rdkit import Chem, RDLogger


from bondnet.dataset.generalized import create_reaction_network_files_and_valid_rows
from bondnet.data.reaction_network import ReactionInNetwork, ReactionNetwork
from bondnet.data.transformers import HeteroGraphFeatureStandardScaler, StandardScaler
from bondnet.data.utils import get_dataset_species, get_hydro_data_functional_groups
from bondnet.utils import to_path, yaml_load, list_split_by_size
from bondnet.data.utils import create_rxn_graph


logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def task_done(future):
    try:
        result = future.result()  # blocks until results are ready
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback)  # traceback of the function


class BaseDataset:
    """
     Base dataset class.

    Args:
     grapher (BaseGraph): grapher object that build different types of graphs:
         `hetero`, `homo_bidirected` and `homo_complete`.
         For hetero graph, atom, bond, and global state are all represented as
         graph nodes. For homo graph, atoms are represented as node and bond are
         represented as graph edges.
     molecules (list or str): rdkit molecules. If a string, it should be the path
         to the sdf file of the molecules.
     labels (list or str): each element is a dict representing the label for a bond,
         molecule or reaction. If a string, it should be the path to the label file.
     extra_features (list or str or None): each element is a dict representing extra
         features provided to the molecules. If a string, it should be the path to the
         feature file. If `None`, features will be calculated only using rdkit.
     feature_transformer (bool): If `True`, standardize the features by subtracting the
         means and then dividing the standard deviations.
     label_transformer (bool): If `True`, standardize the label by subtracting the
         means and then dividing the standard deviations. More explicitly,
         labels are standardized by y' = (y - mean(y))/std(y), the model will be
         trained on this scaled value. However for metric measure (e.g. MAE) we need
         to convert y' back to y, i.e. y = y' * std(y) + mean(y), the model
         prediction is then y^ = y'^ *std(y) + mean(y), where ^ means predictions.
         Then MAE is |y^-y| = |y'^ - y'| *std(y), i.e. we just need to multiple
         standard deviation to get back to the original scale. Similar analysis
         applies to RMSE.
     state_dict_filename (str or None): If `None`, feature mean and std (if
         feature_transformer is True) and label mean and std (if label_transformer is True)
         are computed from the dataset; otherwise, they are read from the file.
    """

    def __init__(
        self,
        grapher,
        molecules,
        labels,
        extra_features=None,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        state_dict_filename=None,
    ):
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"`dtype {dtype}` should be `float32` or `float64`.")

        self.grapher = grapher
        self.molecules = (
            to_path(molecules) if isinstance(molecules, (str, Path)) else molecules
        )
        try:
            self.molecules = [mol.rdkit_mol() for mol in self.molecules]
        except:
            print("molecules already some rdkit object")

        self.raw_labels = to_path(labels) if isinstance(labels, (str, Path)) else labels
        self.extra_features = (
            to_path(extra_features)
            if isinstance(extra_features, (str, Path))
            else extra_features
        )
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.dtype = dtype
        self.state_dict_filename = state_dict_filename

        self.graphs = None
        self.labels = None
        self._feature_size = None
        self._feature_name = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None
        self._label_scaler_mean = None
        self._label_scaler_std = None
        self._species = None
        self._failed = None

        self._load()

    @property
    def feature_size(self):
        """
        Returns a dict of feature size with node type as the key.
        """
        return self._feature_size

    @property
    def feature_name(self):
        """
        Returns a dict of feature name with node type as the key.
        """
        return self._feature_name

    def get_feature_size(self, ntypes):
        """
        Get feature sizes.

        Args:
              ntypes (list of str): types of nodes.

        Returns:
             list: sizes of features corresponding to note types in `ntypes`.
        """
        size = []
        for nt in ntypes:
            for k in self.feature_size:
                if nt in k:
                    size.append(self.feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = f"cannot get feature size for nodes: {ntypes}"
        assert len(ntypes) == len(size), msg

        return size

    @property
    def failed(self):
        """
        Whether an entry (molecule, reaction) fails upon converting using rdkit.

        Returns:
            list of bool: each element indicates whether a entry fails. The size of
                this list is the same as the labels, each one corresponds a label in
                the same order.
            None: is this info is not set
        """
        return self._failed

    def state_dict(self):
        d = {
            "feature_size": self._feature_size,
            "feature_name": self._feature_name,
            "feature_scaler_mean": self._feature_scaler_mean,
            "feature_scaler_std": self._feature_scaler_std,
            "label_scaler_mean": self._label_scaler_mean,
            "label_scaler_std": self._label_scaler_std,
            "species": self._species,
        }

        return d

    def load_state_dict(self, d):
        self._feature_size = d["feature_size"]
        self._feature_name = d["feature_name"]
        self._feature_scaler_mean = d["feature_scaler_mean"]
        self._feature_scaler_std = d["feature_scaler_std"]
        self._label_scaler_mean = d["label_scaler_mean"]
        self._label_scaler_std = d["label_scaler_std"]
        self._species = d["species"]

    def _load(self):
        """Read data from files and then featurize."""
        raise NotImplementedError

    @staticmethod
    def get_molecules(molecules):
        if isinstance(molecules, Path):
            path = str(molecules)
            supp = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
            molecules = [m for m in supp]
        return molecules

    @staticmethod
    def build_graphs(grapher, molecules, features, species):
        """
        Build DGL graphs using grapher for the molecules.

        Args:
            grapher (Grapher): grapher object to create DGL graphs
            molecules (list): rdkit molecules
            features (list): each element is a dict of extra features for a molecule
            species (list): chemical species (str) in all molecules

        Returns:
            list: DGL graphs
        """
        graphs = []
        """
        with ProcessPool(max_workers=12, max_tasks=10) as pool:
            for i, (m, feats) in enumerate(zip(molecules, features)):
                if m is not None:
                    future = pool.schedule(grapher.build_graph_and_featurize, 
                                            args=[m], timeout=30,
                                            kwargs={"extra_feats_info":feats, 
                                                    "dataset_species":species}
                                            )
                    future.add_done_callback(task_done)
                    try:
                        g = future.result()
                        g.graph_id = i
                        graphs.append(g)
                    except:
                        pass
                else: graphs.append(None)

        """
        for i, (m, feats) in tqdm(enumerate(zip(molecules, features))):
            if m is not None:
                g = grapher.build_graph_and_featurize(
                    m, extra_feats_info=feats, element_set=species
                )
                # add this for check purpose; some entries in the sdf file may fail
                g.graph_id = i

            else:
                g = None
            graphs.append(g)

        return graphs

    def __getitem__(self, item):
        """Get data point with index

        Args:
            item (int): data point index

        Returns:
            g (DGLGraph or DGLHeteroGraph): graph ith data point
            lb (dict): Labels of the data point
        """
        (
            g,
            lb,
        ) = (
            self.graphs[item],
            self.labels[item],
        )
        return g, lb

    def __len__(self):
        """
        Returns:
            int: length of dataset
        """
        return len(self.graphs)

    def __repr__(self):
        rst = "Dataset " + self.__class__.__name__ + "\n"
        rst += "Length: {}\n".format(len(self))
        for ft, sz in self.feature_size.items():
            rst += "Feature: {}, size: {}\n".format(ft, sz)
        for ft, nm in self.feature_name.items():
            rst += "Feature: {}, name: {}\n".format(ft, nm)
        return rst


class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        self.dtype = dataset.dtype
        self.dataset = dataset
        self.indices = indices

    @property
    def feature_size(self):
        return self.dataset.feature_size

    @property
    def feature_name(self):
        return self.dataset.feature_name

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ReactionNetworkDatasetGraphs(BaseDataset):
    def __init__(
        self,
        grapher,
        file,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        target="ts",
        filter_species=[2, 3],
        filter_outliers=True,
        filter_sparse_rxns=False,
        feature_filter=False,
        classifier=False,
        debug=False,
        classif_categories=None,
        extra_keys=None,
        dataset_atom_types=None,
        extra_info=None,
        species=["C", "F", "H", "N", "O", "Mg", "Li", "S", "Cl", "P", "O", "Br"],
    ):
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"`dtype {dtype}` should be `float32` or `float64`.")
        self.grapher = grapher
        (
            all_mols,
            all_labels,
            features,
        ) = create_reaction_network_files_and_valid_rows(
            file,
            bond_map_filter=False,
            target=target,
            filter_species=filter_species,
            classifier=classifier,
            debug=debug,
            filter_outliers=filter_outliers,
            categories=classif_categories,
            filter_sparse_rxn=filter_sparse_rxns,
            feature_filter=feature_filter,
            extra_keys=extra_keys,
            extra_info=extra_info,
        )

        self.molecules = all_mols
        self.raw_labels = all_labels
        self.extra_features = features
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.dtype = dtype
        self.state_dict_filename = None
        self.graphs = None
        self.labels = None
        self.target = target
        self.extra_keys = extra_keys
        self._feature_size = None
        self._feature_name = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None
        self._label_scaler_mean = None
        self._label_scaler_std = None
        self._species = species
        self._elements = dataset_atom_types
        self._failed = None
        self.classifier = classifier
        self.classif_categories = classif_categories
        self._load()

    def _load(self):
        logger.info("Start loading dataset")

        # get molecules, labels, and extra features
        molecules = self.get_molecules(self.molecules)
        raw_labels = self.get_labels(self.raw_labels)
        if self.extra_features is not None:
            extra_features = self.get_features(self.extra_features)
        else:
            extra_features = [None] * len(molecules)

        # get state info
        if self.state_dict_filename is not None:
            logger.info(f"Load dataset state dict from: {self.state_dict_filename}")
            state_dict = torch.load(str(self.state_dict_filename))
            self.load_state_dict(state_dict)

        # get species
        # species = get_dataset_species_from_json(self.pandas_df)
        if self._species is None:
            system_species = set()
            for mol in self.molecules:
                species = list(set(mol.species))
                system_species.update(species)

            self._species = sorted(system_species)
        # self._species

        # create dgl graphs
        print("constructing graphs & features....")

        graphs = self.build_graphs(
            self.grapher, self.molecules, extra_features, self._species
        )
        graphs_not_none_indices = [i for i, g in enumerate(graphs) if g is not None]
        print("number of graphs valid: " + str(len(graphs_not_none_indices)))
        print("number of graphs: " + str(len(graphs)))
        assert len(graphs_not_none_indices) == len(
            graphs
        ), "Some graphs are invalid in construction, this should not happen"
        # store feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # feature transformers
        if self.feature_transformer:
            if self.state_dict_filename is None:
                feature_scaler = HeteroGraphFeatureStandardScaler(mean=None, std=None)
            else:
                assert (
                    self._feature_scaler_mean is not None
                ), "Corrupted state_dict file, `feature_scaler_mean` not found"
                assert (
                    self._feature_scaler_std is not None
                ), "Corrupted state_dict file, `feature_scaler_std` not found"

                feature_scaler = HeteroGraphFeatureStandardScaler(
                    mean=self._feature_scaler_mean, std=self._feature_scaler_std
                )

            graphs_not_none = [graphs[i] for i in graphs_not_none_indices]
            graphs_not_none = feature_scaler(graphs_not_none)
            molecules_ordered = [self.molecules[i] for i in graphs_not_none_indices]
            molecules_final = [0 for i in graphs_not_none_indices]
            # update graphs
            for i, g in zip(graphs_not_none_indices, graphs_not_none):
                molecules_final[i] = molecules_ordered[i]
                graphs[i] = g
            self.molecules_ordered = molecules_final

            # if self.device != None:
            #    graph_temp = []
            #    for g in graphs:
            #        graph_temp.append(g.to(self.device))
            #    graphs = graph_temp

            if self.state_dict_filename is None:
                self._feature_scaler_mean = feature_scaler.mean
                self._feature_scaler_std = feature_scaler.std

            logger.info(f"Feature scaler mean: {self._feature_scaler_mean}")
            logger.info(f"Feature scaler std: {self._feature_scaler_std}")

        # create reaction
        reactions = []
        self.labels = []
        self._failed = []
        for i, lb in enumerate(raw_labels):
            mol_ids = lb["reactants"] + lb["products"]
            for d in mol_ids:
                # ignore reaction whose reactants or products molecule is None
                if d not in graphs_not_none_indices:
                    self._failed.append(True)
                    break
            else:
                rxn = ReactionInNetwork(
                    reactants=lb["reactants"],
                    products=lb["products"],
                    atom_mapping=lb["atom_mapping"],
                    bond_mapping=lb["bond_mapping"],
                    total_bonds=lb["total_bonds"],
                    total_atoms=lb["total_atoms"],
                    id=lb["id"],
                    extra_info=lb["extra_info"],
                )

                reactions.append(rxn)
                if "environment" in lb:
                    environemnt = lb["environment"]
                else:
                    environemnt = None

                if self.classifier:
                    lab_temp = torch.zeros(self.classif_categories)
                    lab_temp[int(lb["value"][0])] = 1

                    if lb["value_rev"] != None:
                        lab_temp_rev = torch.zeros(self.classif_categories)
                        lab_temp[int(lb["value_rev"][0])] = 1
                    else:
                        lab_temp_rev = None

                    label = {
                        "value": lab_temp,
                        "value_rev": lab_temp_rev,
                        "id": lb["id"],
                        "environment": environemnt,
                        "atom_map": lb["atom_mapping"],
                        "bond_map": lb["bond_mapping"],
                        "total_bonds": lb["total_bonds"],
                        "total_atoms": lb["total_atoms"],
                        "reaction_type": lb["reaction_type"],
                        "extra_info": lb["extra_info"],
                    }
                    self.labels.append(label)
                else:
                    label = {
                        "value": torch.tensor(
                            lb["value"], dtype=getattr(torch, self.dtype)
                        ),
                        "value_rev": torch.tensor(
                            lb["value_rev"], dtype=getattr(torch, self.dtype)
                        ),
                        "id": lb["id"],
                        "environment": environemnt,
                        "atom_map": lb["atom_mapping"],
                        "bond_map": lb["bond_mapping"],
                        "total_bonds": lb["total_bonds"],
                        "total_atoms": lb["total_atoms"],
                        "reaction_type": lb["reaction_type"],
                        "extra_info": lb["extra_info"],
                    }
                    self.labels.append(label)

                self._failed.append(False)

        self.reaction_ids = list(range(len(reactions)))

        # create reaction network
        self.reaction_network = ReactionNetwork(
            molecules=graphs, reactions=reactions, wrappers=molecules_final
        )
        self.graphs = graphs

        # feature transformers
        if self.label_transformer:
            # normalization
            values = torch.stack([lb["value"] for lb in self.labels])  # 1D tensor
            values_rev = torch.stack([lb["value_rev"] for lb in self.labels])

            if self.state_dict_filename is None:
                mean = torch.mean(values)
                std = torch.std(values)
                self._label_scaler_mean = mean
                self._label_scaler_std = std
            else:
                assert (
                    self._label_scaler_mean is not None
                ), "Corrupted state_dict file, `label_scaler_mean` not found"
                assert (
                    self._label_scaler_std is not None
                ), "Corrupted state_dict file, `label_scaler_std` not found"
                mean = self._label_scaler_mean
                std = self._label_scaler_std

            values = (values - mean) / std
            value_rev_scaled = (values_rev - mean) / std

            # update label
            for i, lb in enumerate(values):
                self.labels[i]["value"] = lb
                self.labels[i]["scaler_mean"] = mean
                self.labels[i]["scaler_stdev"] = std
                self.labels[i]["value_rev"] = value_rev_scaled[i]

            logger.info(f"Label scaler mean: {mean}")
            logger.info(f"Label scaler std: {std}")

        logger.info(f"Finish loading {len(self.labels)} reactions...")

    @staticmethod
    def build_graphs(grapher, molecules, features, species):
        """
        Build DGL graphs using grapher for the molecules.

        Args:
            grapher (Grapher): grapher object to create DGL graphs
            molecules (list): rdkit molecules
            features (list): each element is a dict of extra features for a molecule
            species (list): chemical species (str) in all molecules

        Returns:
            list: DGL graphs
        """

        count = 0
        graphs = []

        for ind, mol in enumerate(molecules):
            feats = features[count]
            if mol is not None:
                g = grapher.build_graph_and_featurize(
                    mol, extra_feats_info=feats, element_set=species
                )

                # add this for check purpose; some entries in the sdf file may fail
                g.graph_id = ind
            else:
                g = None
            graphs.append(g)
            count += 1

        return graphs

    @staticmethod
    def get_labels(labels):
        if isinstance(labels, Path):
            labels = yaml_load(labels)
        return labels

    @staticmethod
    def get_features(features):
        if isinstance(features, Path):
            features = yaml_load(features)
        return features

    def __getitem__(self, item):
        rn, rxn, lb = self.reaction_network, self.reaction_ids[item], self.labels[item]
        # reactions, graphs = rn.subselect_reactions([self.reaction_ids[item]])
        # return rn, rxn, lb, reactions, graphs
        return rn, rxn, lb

    """    def __getitem__(self, item):
        rn, rxn_ids, lb = (
            self.reaction_network,
            self.reaction_ids[item],
            self.labels[item],
        )
        reactions, graphs = rn.subselect_reactions([rxn_ids])
        return reactions, graphs, lb
    """

    def __len__(self):
        return len(self.reaction_ids)


def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=None):
    """
    Split a dataset into training, validation, and test set.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(size)
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train + num_val]
    test_idx = idx[num_train + num_val :]
    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]

