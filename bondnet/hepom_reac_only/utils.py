try:
    from rdkit.Chem import rdEHTTools  # requires RDKit 2019.9.1 or later
except ImportError:
    rdEHTTools = None
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, Get3DDistanceMatrix

import os
import time
import pickle
import yaml
import random
import torch
import torch_geometric
import logging
import warnings
import sys
import shutil
import itertools
import copy
from pathlib import Path
import numpy as np
from typing import List, Any
from collections import defaultdict
import networkx as nx
import git




global __ATOM_LIST__
__ATOM_LIST__ = [
    "h",
    "he",
    "li",
    "be",
    "b",
    "c",
    "n",
    "o",
    "f",
    "ne",
    "na",
    "mg",
    "al",
    "si",
    "p",
    "s",
    "cl",
    "ar",
    "k",
    "ca",
    "sc",
    "ti",
    "v ",
    "cr",
    "mn",
    "fe",
    "co",
    "ni",
    "cu",
    "zn",
    "ga",
    "ge",
    "as",
    "se",
    "br",
    "kr",
    "rb",
    "sr",
    "y",
    "zr",
    "nb",
    "mo",
    "tc",
    "ru",
    "rh",
    "pd",
    "ag",
    "cd",
    "in",
    "sn",
    "sb",
    "te",
    "i",
    "xe",
    "cs",
    "ba",
    "la",
    "ce",
    "pr",
    "nd",
    "pm",
    "sm",
    "eu",
    "gd",
    "tb",
    "dy",
    "ho",
    "er",
    "tm",
    "yb",
    "lu",
    "hf",
    "ta",
    "w",
    "re",
    "os",
    "ir",
    "pt",
    "au",
    "hg",
    "tl",
    "pb",
    "bi",
    "po",
    "at",
    "rn",
    "fr",
    "ra",
    "ac",
    "th",
    "pa",
    "u",
    "np",
    "pu",
]


global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[3] = [1, 2, 3, 4, 5, 6]
atomic_valence[5] = [3, 4]
atomic_valence[6] = [4]
atomic_valence[7] = [3, 4]
atomic_valence[8] = [2, 1, 3]
atomic_valence[9] = [1]
atomic_valence[12] = [3, 5, 6]
atomic_valence[14] = [4]
atomic_valence[15] = [5, 3]  # [5,4,3]
atomic_valence[16] = [6, 3, 2]  # [6,4,2]
atomic_valence[17] = [1]
atomic_valence[32] = [4]
atomic_valence[35] = [1]
atomic_valence[53] = [1]

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[3] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[12] = 2
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7
logger = logging.getLogger(__name__)

def to_path(path):
    return Path(path).expanduser().resolve()

def yaml_load(filename):
    with open(to_path(filename), "r") as f:
        obj = yaml.safe_load(f)
    return obj

def list_split_by_size(data: List[Any], sizes: List[int]) -> List[List[Any]]:
    """
    Split a list into `len(sizes)` chunks with the size of each chunk given by `sizes`.
    This is a similar to `np_split_by_size` for a list. We cannot use
    `np_split_by_size` for a list of graphs, because DGL errors out if we convert a
    list of graphs to an array of graphs.
    Args:
        data: the list of data to split
        sizes: size of each chunk.
    Returns:
        a list of list, where the size of each inner list is given by `sizes`.
    Example:
        >>> list_split_by_size([0,1,2,3,4,5], [1,2,3])
        >>>[[0], [1,2], [3,4,5]]
    """
    assert len(data) == sum(
        sizes
    ), f"Expect len(array) be equal to sum(sizes); got {len(data)} and {sum(sizes)}"

    indices = list(itertools.accumulate(sizes))

    new_data = []
    a = []
    for i, x in enumerate(data):
        a.append(x)
        if i + 1 in indices:
            new_data.append(a)
            a = []

    return new_data

def get_git_repo():
    repo = git.Repo('.', search_parent_directories=True)
    repo_path = Path(repo.working_tree_dir)

    return repo_path

