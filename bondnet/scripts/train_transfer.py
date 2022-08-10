import time, wandb, torch
import numpy as np 
from tqdm import tqdm

from torchmetrics import R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bondnet.model.metric import EarlyStopping
from bondnet.data.dataset import ReactionNetworkDatasetGraphs
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.featurizer import (
    AtomFeaturizerGraph,
    BondAsNodeGraphFeaturizer,
    GlobalFeaturizerGraph,
)
from bondnet.data.grapher import (
    HeteroCompleteGraphFromDGLAndPandas,
)
from bondnet.data.dataset import train_validation_test_split
#from bondnet.scripts.create_label_file import read_input_files
#from bondnet.model.metric import WeightedL1Loss, WeightedMSELoss
from bondnet.utils import seed_torch, pickle_dump, parse_settings
from bondnet.model.training_utils import (
    evaluate, 
    evaluate_classifier, 
    train, 
    train_classifier, 
    load_model
)
seed_torch()

def evaluate_r2(model, nodes, data_loader):
    model.eval()
    with torch.no_grad():
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            
            pred = model(batched_graph, feats, label["reaction"])
            pred = pred.view(-1)
            target = target.view(-1)

    r2_call = R2Score()
    r2 = r2_call(pred, target)
    return r2


def get_grapher():

    atom_featurizer = AtomFeaturizerGraph()
    bond_featurizer = BondAsNodeGraphFeaturizer()
    global_featurizer = GlobalFeaturizerGraph(allowed_charges=[-2, -1, 0, 1])
    grapher = HeteroCompleteGraphFromDGLAndPandas(
        atom_featurizer, bond_featurizer, global_featurizer
    )
    return grapher


def main():
    best = 1e10
    feature_names = ["atom", "bond", "global"]
    path_mg_data = "../dataset/mg_dataset/"
    dict_train = parse_settings()
    path_mg_data = "../../../dataset/mg_dataset/20220613_reaction_data.json"
        
    if(dict_train["classifier"]):
        classif_categories = 5 # update this later
        wandb.init(project="project_classification_test")
    else:
        classif_categories = None
        wandb.init(project="project_regression_test")
    wandb.config.update(dict_train)

    if dict_train["on_gpu"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_train["gpu"] = device
        model.to(device)
    else:
        dict_train["gpu"] = "cpu"
    print("train on device: {}".format(dict_train["gpu"]))

    dataset = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(), 
        file=path_mg_data, 
        out_file="./", 
        target = 'ts', 
        classifier = dict_train["classifier"], 
        classif_categories=classif_categories, 
        debug = dict_train["debug"]
    )
    
    dict_train['in_feats'] = dataset.feature_size
    model, optimizer = load_model(dict_train)

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.15, test=0.15
    )
    
    train_loader = DataLoaderReactionNetwork(trainset, batch_size=dict_train['batch_size'], 
    shuffle=True)
    val_loader = DataLoaderReactionNetwork(
        valset, batch_size=len(valset), shuffle=False
    )
    test_loader = DataLoaderReactionNetwork(
        testset, batch_size=len(testset), shuffle=False
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=25, verbose=True)
    stopper = EarlyStopping(patience=150)
    stopper_transfer = EarlyStopping(patience=150)

    if(dict_train['transfer']):
        dataset_transfer = ReactionNetworkDatasetGraphs(
        grapher=get_grapher(), file=path_mg_data, out_file="./", 
        target = 'diff', 
        classifier = dict_train["classifier"], 
        classif_categories=classif_categories, 
        debug = dict_train["debug"]
        )

        trainset_transfer, valset_tranfer, _ = train_validation_test_split(
        dataset_transfer, validation=0.15, test=0.01
        )
        dataset_transfer_loader = DataLoaderReactionNetwork(
            trainset_transfer, batch_size=dict_train['batch_size'], 
        shuffle=True)
        dataset_transfer_loader_val = DataLoaderReactionNetwork(
            valset_tranfer, batch_size=dict_train['batch_size'], 
        shuffle=True)

        print("Initiating Training w/ transfer...")
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of Trainable Model Params: {}".format(params))
        
        
        for epoch in tqdm(range(dict_train['transfer_epochs'])):
            if(dict_train["classifier"]):
                _, _ = train_classifier(
                    model, 
                    feature_names, 
                    dataset_transfer_loader,
                    optimizer, 
                    device = dict_train["gpu"], 
                    categories = classif_categories
                )
                val_acc_transfer, f1_score = evaluate_classifier(
                    model, 
                    feature_names, 
                    dataset_transfer_loader_val, 
                    device = dict_train["gpu"],
                    categories = classif_categories
                )
            else:
                _, _ = train(
                    model, 
                    feature_names, 
                    dataset_transfer_loader, 
                    optimizer, 
                    device = dict_train["gpu"]
                )
                val_acc_transfer = evaluate(
                    model, 
                    feature_names, 
                    dataset_transfer_loader_val, 
                    device = dict_train["gpu"]
                )

            if stopper_transfer.step(val_acc_transfer):
                break

        # freeze model layers but fc
        model.gated_layers.requires_grad_(False)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Freezing Gated Layers....")
        print("Number of Trainable Model Params: {}".format(params))

    t1 = time.time()
    # optimizer, loss function and metric function
    # main training loop
    if(dict_train["classifier"]):
        print("# Epoch     Loss         TrainAcc        ValAcc        ValF1")
    else: 
        print("# Epoch     Loss         TrainAcc        ValAcc        ValR2")        
    
    for epoch in range(dict_train['epochs']):
        # train on training set
        if(dict_train["classifier"]):
            loss, train_acc = train_classifier(
                model, 
                feature_names, 
                train_loader, 
                optimizer, 
                device = dict_train["gpu"],
                categories = classif_categories
            )

            # evaluate on validation set
            val_acc, f1_score = evaluate_classifier(
                model, 
                feature_names, 
                val_loader, 
                device = dict_train["gpu"],
                categories = classif_categories
            )

            wandb.log({"acc validation": val_acc})
            wandb.log({"f1 validation": f1_score})
            print(
                "{:5d}   {:12.6e}   {:12.2e}   {:12.6e}   {:.2f}".format(
                    epoch, loss, train_acc, val_acc, f1_score
                )
            )
            
        else: 
            loss, train_acc = train(
            model, 
            feature_names, 
            train_loader, 
            optimizer, 
            dict_train["gpu"]
            )
            # evaluate on validation set
            val_acc = evaluate(model, feature_names, val_loader, dict_train["gpu"])
            val_r2 = evaluate_r2(model, feature_names, val_loader)
        
            wandb.log({"loss": loss})
            wandb.log({"acc_val": val_acc})
            wandb.log({"r2_val": val_r2})

            print(
                "{:5d}   {:12.6e}   {:12.2e}   {:12.6e}   {:.2f}".format(
                    epoch, loss, train_acc, val_acc, val_r2
                )
            )

        # save checkpoint for best performing model
        is_best = val_acc < best
        if is_best:
            best = val_acc
            torch.save(model.state_dict(), "checkpoint.pkl")

        if(dict_train["early_stop"]):
            if stopper.step(val_acc):
                pickle_dump(
                    best, dict_train["save_hyper_params"]
                )  # save results for hyperparam tune
                break
        scheduler.step(val_acc)

    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint)
    
    if(dict_train["classifier"]):
        test_acc, test_f1 = evaluate_classifier(
            model, 
            feature_names, 
            test_loader, 
            device = dict_train["gpu"],
            categories = classif_categories
        )

        wandb.log({"acc validation": test_acc})
        wandb.log({"f1 validation": test_f1})
        print("Test Acc: {:12.6e}".format(test_acc))
        print("Test F1: {:12.6e}".format(test_f1))


    else: 
        test_acc = evaluate(model, feature_names, test_loader)
        wandb.log({"loss_test": test_acc})
        print("TestAcc: {:12.6e}".format(test_acc))
    
    t2 = time.time()
    print("Time to Training: {:5.1f} seconds".format(float(t2 - t1)))

main()
