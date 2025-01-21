import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cs236781.train_results import FitResult

from .cnn import CNN, ResNet
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
}


def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======

    # Create a BinaryClassifier model
    dims = [width]*depth + [2]
    nonlins=['relu', *['tanh']*(depth-1), 'none']
    mlp_model = MLP(in_dim=2, dims=dims,nonlins=nonlins)
    model = BinaryClassifier(model=mlp_model, threshold=0.5)

    # Train the model
    loss_func = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.015, 0.015, 0.92
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    trainer = ClassifierTrainer(model, loss_func, optimizer)

    # training the model
    valid_acc = trainer.fit(dl_train=dl_train, dl_test=dl_valid, num_epochs=n_epochs, print_every=0).test_acc[-1]
    
    # select threshold
    thresh = select_roc_thresh(model, *dl_valid.dataset.tensors, plot=False)

    # Setting optimal threshold
    model.threshold = thresh

    # evaluating one epoch on the test set
    test_acc = trainer.test_epoch(dl_test, verbose=False).accuracy
    
    # ========================
    return model, thresh, valid_acc, test_acc


def cnn_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=2,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    **kw,
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======

    # init parameters 
    # TODO: there has to be a smarter way to do this.... check with Murad about **kw use
    if 'conv_params' not in kw.keys():
        conv_params = {'kernel_size': 3, 'padding': 1, 'stride': 1}
    else:
        conv_params = kw['conv_params']

    if 'activation_type' not in kw.keys():
        activation_type = 'relu'
    else:
        activation_type = kw['activation_type']

    if 'activation_params' not in kw.keys():
        activation_params = {}
    else:
        activation_params = kw['activation_params']    

    if 'pooling_type' not in kw.keys():
        pooling_type = 'max'
    else:
        pooling_type = kw['pooling_type']  

    if 'pooling_params' not in kw.keys():
        pooling_params = {'kernel_size': 2}
    else:
        pooling_params = kw['pooling_params']

    if 'print_every' not in kw.keys():
        print_every = 0
    else:
        print_every = kw['print_every']       

    channels = filters_per_layer * layers_per_block

    in_size = ds_train[0][0].shape

    # load data to torch data loader format
    dl_train = DataLoader(dataset=ds_train, batch_size=bs_train)
    dl_test = DataLoader(dataset=ds_test, batch_size=bs_test)

    # initialize model
    model = model_cls(in_size= in_size, out_classes= 10, channels= channels, pool_every = pool_every, hidden_dims = hidden_dims, 
                conv_params = conv_params, activation_type = activation_type, activation_params =activation_params, 
                pooling_type = pooling_type, pooling_params = pooling_params).to(device)


    # define classifier #TODO: check ret type with Murad
    classifier = ArgMaxClassifier(model).to(device)

    # define a loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # initialize an optimizer
    optimizer = torch.optim.Adam(params= classifier.parameters(), lr= lr, weight_decay= reg)

    # initialize a trainer
    trainer = ClassifierTrainer(model= classifier, loss_fn= loss_fn, optimizer= optimizer, device= device)


    # TODO: tidy up the print every thing using **kw
    fit_res = trainer.fit(dl_train= dl_train, dl_test= dl_test, num_epochs= epochs, checkpoints= checkpoints, early_stopping= early_stopping,
                print_every= print_every)

    # raise NotImplementedError()
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
