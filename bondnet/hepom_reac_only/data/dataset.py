import torch, time, itertools, logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from tqdm import tqdm
from rdkit import Chem, RDLogger
from bondnet.hepom_reac_only.data.utils import get_dataset_species, get_rdkit_mols_from_path, get_rdkit_mols_from_list, get_aromatic_label
from bondnet.hepom_reac_only.utils import to_path, yaml_load, list_split_by_size
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def task_done(future):
    try:
        result = future.result() # blocks until results are ready
    except TimeoutError as error:
        print(f"Function took longer than {error.args[1]} seconds")
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback) #traceback of the functiom


class BaseDataset:
    """
    Base dataset class.
    Args:
     grapher (BaseGraph): pyg grapher object that build different types of graphs:
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

        self.labels = to_path(labels) if isinstance(labels, (str, Path)) else labels
        self.extra_features = (
            to_path(extra_features)
            if isinstance(extra_features, (str,Path))
            else extra_features
        )

        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.state_dict_filename = state_dict_filename

        self.graphs = None
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
        Returns a dict of feature size with node type as the key
        """
        return self._feature_size
        
    @property
    def feature_name(self):
        """
        Returns a dict of feature names with node type as the key
        """
        return self._feature_name
        
    def get_feature_size(self, ntypes):
        """
        Get feature sizes.

        Args:
            ntypes (list of str): types of nodes.

        Returns:
            list: sizes of features corresponding to node types in `ntypes`.
        """
        size = []
        for nt in ntypes:
            for k in self.feature_size:
                if nt in k:
                    size.append(self.feature_size[k])
        msg = f"cannot get feature size for nodes: {ntypes}"
        assert len(ntypes) == len(size), msg

        return size
        
    @property
    def failed(self):
        """
        Whether an entry (molecule) fails upon converting using rdkit.

        Returns:
            list of bool: each element indicates whether a entry fails to get converted. The size of
            this list is the same as the size of the labels, each one corresponds a molecule entry in
            the same order.

            None: if this info is not set
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
            "species": self._species
        }

        return d
        
    def load_state_dict(self, d):
        self._feature_size = d['feature_size']
        self._feature_name = d['feature_name']
        self._feature_scaler_mean = d['feature_scaler_mean']
        self._feature_scaler_std = d['feature_scaler_std']
        self._label_scaler_mean = d['label_scaler_mean']
        self._label_scaler_std = d['label_scaler_std']
        self._species = d['species']

    def _load(self):
        """
        Read data from files and then featurize
        """
        raise NotImplementedError
        
    @staticmethod
    def get_rdkit_molecules(molecules):
        if isinstance(molecules, Path):
            path = str(molecules)
            molecules = get_rdkit_mols_from_path(path)
        elif isinstance(molecules, list):
            molecules = get_rdkit_mols_from_list(molecules)

        return molecules
    

        
    @staticmethod
    def build_graphs_and_get_labels(grapher, molecules, features, species, labels):
        """
        Building PyG graphs using grapher for the rdkit molecules

        Args:
            grapher (Grapher): grapher object to create PyG graphs
            molecules (list): rdkit molecules
            features (list): each element is a dict of extra features for a molecule
            species (list): chemical species (str) in all molecules

        Returns:
            list: PyG graphs
        """

        graphs = []
        temp_labels = labels
        labels = []



        for i, (m,feats) in tqdm(enumerate(zip(molecules, features))):
            if m is not None:
                g = grapher.build_graph_and_featurize(
                    m, extra_feats_info = feats, dataset_species=species
                )
                if (temp_labels == None):
                    lb = get_aromatic_label(m)
                elif isinstance(temp_labels, list):
                    lb = temp_labels[i]
                else:
                    print("Either pass labels as None or a list")
                g.graph_id = i
            else:
                g = None
            graphs.append(g)
            labels.append(lb)
        return graphs, labels
        
    def __getitem__(self, item):
        """
        Get a graph data point with its label
        
        Args:
            item (int): data point index
        
        Returns:
            g (PyG Graph): graph ith data point
            lb (int): label for the data point
        """

        g, lb = (
            self.graphs[item],
            self.labels[item],
        )

    def __len__(self):
        """
        Returns:
            int: length of dataset
        """
        return(len(self.graphs))
    
    def __repr__(self):
        rst = "Dataset " + self.__class__.__name__ + "\n"
        rst += f"Length : {len(self)}"
        for ft, sz in self.feature_size.items():
            rst += f"Feature: {ft}, size: {sz}"
        for ft, nm in self.feature_name.items():
            rst += f"Feature: {ft}, size: {nm}"
        return rst
        
class MoleculeDataset(BaseDataset):
    def __init__(
        self,
        grapher,
        molecules,
        labels,
        extra_features=None,
        feature_transformer=None,
        label_transformer=None,
        dtype='float32',
    ):
        super().__init__(
            grapher=grapher,
            molecules=molecules,
            labels=labels,
            extra_features=extra_features,
            feature_transformer=feature_transformer,
            label_transformer=label_transformer,
            dtype=dtype,
        )
    
    def _load(self):

        logger.info("Start loading dataset")
        molecules = self.get_rdkit_molecules(self.molecules)
        species = get_dataset_species(molecules)
        aroma_flag = 0
        if self.extra_features is not None:
            features = yaml_load(self.extra_features)
        else:
            features = [None] * len(molecules)

        self.graphs = []
        if (self.labels == None):
            aroma_flag = 1
            self.labels = []
        else:
            temp_lbs = self.labels
            self.labels = []
        
        print(aroma_flag)

        for i, (mol,feats) in enumerate(zip(molecules,features)):

            if i%100 == 0:
                logger.info(f"Processing molecule {i}/{len(molecules)}")
            
            if mol is None:
                continue

            if (aroma_flag == 1):
                lb = get_aromatic_label(mol)
                #lb = torch.tensor(lb, dtype=getattr(torch, self.dtype))
                lb = torch.tensor(lb, dtype=torch.long)

                self.labels.append({"value": lb, "id": i})
            else:
                lb = temp_lbs[i]
                lb = torch.tensor(lb, dtype=torch.float)
                self.labels.append({"value": lb, "id": i})


            # get graph and label
            g = self.grapher.build_graph_and_featurize(
                mol, extra_feats_info = feats, dataset_species = species
            )

            g.graph_id = i
            g.y = lb
            self.graphs.append(g)

        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info(f'Feature name: {self.feature_name}')
        logger.info(f'Feature size: {self.feature_size}')

class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        #self.dtype = dataset.dtype
        self.dataset = dataset
        self.indices = indices

    @property
    def feature_size(self):
        return self.dataset.feature_size
    
    @property
    def feature_name(self):
        return self.dataset.feature_name
    
    @property
    def num_classes(self):
        return self.dataset.num_classes
    
    def __getitem__(self, idx):
        return self.dataset.graphs[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)
    
def train_validation_test_split(dataset, validation = 0.1, test = 0.1, random_seed = None):
    """
    Split a dataset into training, validation, and test set.

    The training will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test set.
            Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test set] 
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
        Subset(dataset, test_idx)
    ]