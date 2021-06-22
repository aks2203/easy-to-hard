""" utils.py
    utility functions and classes
    Developed as part of Easy-To-Hard project
    April 2021
"""

import datetime
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass

import einops
from easy_to_hard_data import ChessPuzzleDataset
import torch
import torch.utils.data as data
from icecream import ic
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
from icecream import ic

from models.feed_forward_net import ff_net
from models.recurrent_net import recur_net


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def get_dataloaders(train_batch_size, test_batch_size, eval_start=600000, eval_end=700000, shuffle=True):
    """ Function to get pytorch dataloader objects
    input:
        train_batch_size:   int, Size of mini batches for training
        test_batch_size:    int, Size of mini batches for testing
        eval_start:         int, Which index does the eval set start at
        eval_end:           int, Which index does the eval set end at
        shuffle:            bool, Data shuffle switch
    return:
        trainloader:    Pytorch dataloader object with training data
        testloader:     Pytorch dataloader object with testing data
        evalloader:     Pytorch dataloader object with eval (harder) data
    """

    dataset = ChessPuzzleDataset("./data", train=True, download=True)
    evalset = ChessPuzzleDataset("./data", idx_start=eval_start, idx_end=eval_end, download=True)

    train_split = int(5 / 6 * len(dataset))
    train_data, test_data = torch.utils.data.random_split(dataset,
                                                          [train_split, len(dataset) - train_split],
                                                          generator=torch.Generator().manual_seed(42))
    trainloader = data.DataLoader(train_data, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(test_data, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    evalloader = data.DataLoader(evalset, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    return trainloader, testloader, evalloader


def get_model(model, width, depth):
    """Function to load the model object
    input:
        model:      str, Name of the model
        width:      int, Width of network
        depth:      int, Depth of network
    return:
        net:        Pytorch Network Object
    """
    model = model.lower()
    net = eval(model)(depth=depth, width=width)
    return net


def get_optimizer(optimizer_name, net, lr):
    optimizer_name = optimizer_name.lower()

    base_params = [p for n, p in net.named_parameters()]
    recur_params = []
    iters = 1

    all_params = [{'params': base_params}, {'params': recur_params, 'lr': lr / iters}]

    if optimizer_name == "sgd":
        optimizer = SGD(all_params, lr=lr, weight_decay=2e-4, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=2e-4)
    elif optimizer_name == "adamw":
        optimizer = AdamW(all_params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                          amsgrad=False)
    else:
        print(f"{ic.format()}: Optimizer choise of {optimizer_name} not yet implmented. Exiting.")
        sys.exit()

    return optimizer


def load_model_from_checkpoint(model, model_path, width, depth):
    net = get_model(model, width, depth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    state_dict["net"] = remove_parallel(state_dict["net"])
    net.load_state_dict(state_dict["net"])
    net = net.to(device)

    return net, state_dict["epoch"], state_dict["optimizer"]


def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


@dataclass
class OptimizerWithSched:
    """Attributes for optimizer, lr schedule, and lr warmup"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"


def remove_parallel(state_dict):
    """state_dict: state_dict of model saved with DataParallel()
    returns state_dict without extra module level"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]    # remove module.
        new_state_dict[name] = v
    return new_state_dict


def test(net, testloader, mode, device):
    try:
        accuracy = eval(f"test_{mode}")(net, testloader, device)
    except NameError:
        print(f"{ic.format()}: test_{mode}() not implemented. Exiting.")
        sys.exit()
    return accuracy


def test_default(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            tk = torch.topk(outputs[:, 1].view(-1, 64), 2, dim=1)[0].min(dim=1)[0]
            big_tk = einops.repeat(tk, 'n -> n k', k=8)
            big_tk = einops.repeat(big_tk, 'n m -> n m k', k=8)
            outputs[:, 1][outputs[:, 1] < big_tk] = -float("Inf")
            outputs[:, 0] = -float("Inf")
            predicted = outputs.argmax(1)
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def test_max_conf(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    nm = net.module
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):

            inputs, targets = inputs.to(device), targets.to(device)
            net(inputs)
            confidence_array = torch.zeros(nm.iters, inputs.size(0))
            for i, thought in enumerate(nm.thoughts):
                if i < 8:
                    continue
                thought_s = torch.nn.functional.softmax(thought.detach(), dim=1)
                confidence_array[i] = thought_s.max(dim=1)[0].sum([1, 2])

            exit_iter = confidence_array.argmax(0)
            outputs = nm.thoughts[exit_iter, torch.arange(nm.thoughts.size(1))].squeeze()

            tk = torch.topk(outputs[:, 1].view(-1, 64), 2, dim=1)[0].min(dim=1)[0]
            test = einops.repeat(tk, 'n -> n k', k=8)
            test = einops.repeat(test, 'n m -> n m k', k=8)
            outputs[:, 1][outputs[:, 1] < test] = -float("Inf")
            outputs[:, 0] = -float("Inf")
            predicted = outputs.argmax(1)
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()

            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def to_json(stats, out_dir, log_name="test_stats.json"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    if os.path.isfile(fname):
        with open(fname, 'r') as fp:
            data_from_json = json.load(fp)
            num_entries = data_from_json['num entries']
        data_from_json[num_entries] = stats
        data_from_json["num entries"] += 1
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)
    else:
        data_from_json = {0: stats, "num entries": 1}
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)


def to_log_file(out_dict, out_dir, log_name="log.txt"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, "a") as fh:
        fh.write(str(now()) + " " + str(out_dict) + "\n" + "\n")

    print("logging done in " + out_dir + ".")


def train(net, trainloader, mode, optimizer_obj, device):
    try:
        train_loss, acc = eval(f"train_{mode}")(net, trainloader, optimizer_obj, device)
    except NameError:
        print(f"{ic.format()}: train_{mode}() not implemented. Exiting.")
        sys.exit()
    return train_loss, acc


def train_default(net, trainloader, optimizer_obj, device):
    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup
    criterion = torch.nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(trainloader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device).long()
        optimizer.zero_grad()
        outputs = net(inputs)

        reshaped_outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()

        loss = criterion(reshaped_outputs.view(-1, 2), targets.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*targets.size(0)

        predicted = outputs.argmax(1)
        correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / total
    acc = 100.0 * correct / total
    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc
