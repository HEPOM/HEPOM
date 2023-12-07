import pytorch_lightning as pl
import torch.distributed as dist
import os
from bondnet.data.dataset import (
    ReactionNetworkDatasetGraphs,
    train_validation_test_split,
)

# from bondnet.data.lmdb import LmdbDataset, CRNs2lmdb


from bondnet.data.dataloader import collate_parallel, DataLoaderReactionNetworkParallel
from bondnet.model.training_utils import get_grapher

class BondNetLightningDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prepared = False

    def prepare_data(self):
        if self.prepared:
            return self.entire_dataset._feature_size, self.entire_dataset._feature_name
        
        else:
            self.entire_dataset = ReactionNetworkDatasetGraphs(
                grapher=get_grapher(self.config["model"]["extra_features"]),
                file=self.config["dataset"]["data_dir"],
                target=self.config["dataset"]["target_var"],
                classifier=self.config["model"]["classifier"],
                classif_categories=self.config["model"]["classif_categories"],
                filter_species=self.config["model"]["filter_species"],
                filter_outliers=self.config["model"]["filter_outliers"],
                filter_sparse_rxns=False,
                debug=self.config["model"]["debug"],
                extra_keys=self.config["model"]["extra_features"],
                extra_info=self.config["model"]["extra_info"],
            )

            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = train_validation_test_split(
                self.entire_dataset,
                validation=self.config["optim"]["val_size"],
                test=self.config["optim"]["test_size"],
            )

            # print("done creating lmdb" * 10)
            self.prepared = True
            return self.entire_dataset._feature_size, self.entire_dataset._feature_name

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def train_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.train_ds,
            batch_size=self.config["optim"]["batch_size"],
            shuffle=True,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.test_ds,
            batch_size=len(self.test_ds),
            shuffle=False,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoaderReactionNetworkParallel(
            dataset=self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            collate_fn=collate_parallel,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
        )
