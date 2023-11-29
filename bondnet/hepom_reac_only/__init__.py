__version__ = "0.0.1"

import logging
import os
import torch

logging.basicConfig(
    filename="hepom_reac_only.log",
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    level=logging.INFO,
)

os.environ["DGLBACKEND"] = "pytorch"
os.environ["TORCH"] = torch.__version__