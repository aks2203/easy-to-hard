""" utils.py
    utility functions and classes
    Developed as part of Easy-To-Hard project
    April 2021
"""

from collections import OrderedDict
from dataclasses import dataclass
import datetime
import json
import os
import sys

from easy_to_hard_data import PrefixSumDataset
from icecream import ic
import torch
import torch.utils.data as data
from torch.optim import SGD, Adam, AdamW, Adadelta
from tqdm import tqdm

from models.feed_forward_net import ff_net
from models.recurrent_net import recur_net
from models.recurrent_dilated_net import recur_dilated_net

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def get_dataloaders(train_batch_size, test_batch_size, train_data, eval_data, train_split=0.8, shuffle=True):
    """ Function to get pytorch dataloader objects
    input:
        dataset:            str, Name of the dataset
        train_batch_size:   int, Size of mini batches for training
        test_batch_size:    int, Size of mini batches for testing
        train_data:         int, Number of bits in the training set
        eval_data:          int, Number of bits in the training set
        train_split:        float, Portion of training data to use for training (vs. testing in-distribution)
        shuffle:            bool, Data shuffle switch
    return:
        trainloader:    Pytorch dataloader object with training data
        testloader:     Pytorch dataloader object with testing data
    """
    if train_split >= 1.0 or train_split <= 0:
        print(f"{ic.format()}: Split {train_split} is not between 0 and 1 in "
              f"get_dataloaders(). Exiting.")
        sys.exit()

    dataset = PrefixSumDataset("./data", num_bits=train_data)
    evalset = PrefixSumDataset("./data", num_bits=eval_data)

    num_train = int(train_split * len(dataset))

    trainset, testset = torch.utils.data.random_split(dataset,
                                                      [num_train, int(1e4 - num_train)],
                                                      generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(testset, num_workers=0, batch_size=test_batch_size,
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


def get_optimizer(optimizer_name, model, net, lr):
    optimizer_name = optimizer_name.lower()
    model = model.lower()

    base_params = [p for n, p in net.named_parameters()]
    recur_params = []
    iters = 1

    # if "recur" in model:
    #     base_params = [p for n, p in net.named_parameters() if "recur" not in n]
    #     recur_params = [p for n, p in net.named_parameters() if "recur" in n]
    #     iters = net.iters
    # else:
    #     base_params = [p for n, p in net.named_parameters()]
    #     recur_params = []
    #     iters = 1

    all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]

    if optimizer_name == "sgd":
        optimizer = SGD(all_params, lr=lr, weight_decay=2e-4, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=2e-4)
    elif optimizer_name == "adamw":
        optimizer = AdamW(all_params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                          amsgrad=False)
    elif optimizer_name == "adadelta":
        optimizer = Adadelta(all_params, lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
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
    clip: "typing.Any"


def remove_parallel(state_dict):
    """state_dict: state_dict of model saved with DataParallel()
    returns state_dict without extra module level"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    return new_state_dict


def test(net, testloader, mode, device):
    try:
        accuracy = eval(f"test_{mode}")(net, testloader, device)
    except NameError:
        print(f"{ic.format()}: test_{mode}() not implemented. Exiting.")
        sys.exit()
    return accuracy


def test_bit_wise_per_iter(net, testloader, device):
    net.eval()
    net.to(device)
    total = 0
    nm = net.module

    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(testloader), leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            net(inputs)
            for j, thought in enumerate(nm.thoughts):
                predicted = thought.argmax(1)
                if i == 0 and j == 0:
                    correct = torch.zeros(len(nm.thoughts), inputs.size(-1)).to(device)
                correct[j] += (predicted == targets).sum(0)
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    # print(accuracy)
    return accuracy


def test_bit_wise(net, testloader, device):
    net.eval()
    net.to(device)
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(testloader), leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            predicted = outputs.argmax(1)
            if i == 0:
                correct = (predicted == targets).sum(0)
            else:
                correct += (predicted == targets).sum(0)
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    # print(accuracy)
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

            predicted = outputs.argmax(1)
            correct += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def test_max_conf(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    softmax = torch.nn.functional.softmax
    nm = net.module

    with torch.no_grad():
        for inputs, targets in testloader:

            inputs, targets = inputs.to(device), targets.to(device)
            net(inputs)
            confidence_array = torch.zeros(nm.iters, inputs.size(0))
            for i, thought in enumerate(nm.thoughts):
                conf = softmax(thought.detach(), dim=1).max(1)[0]
                confidence_array[i] = conf.sum([1])

            exit_iter = confidence_array.argmax(0)
            best_thoughts = nm.thoughts[exit_iter, torch.arange(nm.thoughts.size(1))].squeeze()
            if best_thoughts.shape[0] != inputs.shape[0]:
                best_thoughts = best_thoughts.unsqueeze(0)
            predicted = best_thoughts.argmax(1)
            correct += torch.amin(predicted == targets, dim=[1]).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def to_json(stats, out_dir, log_name="test_stats.json"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    if os.path.isfile(fname):
        with open(fname, "r") as fp:
            data_from_json = json.load(fp)
            num_entries = data_from_json["num entries"]
        data_from_json[num_entries] = stats
        data_from_json["num entries"] += 1
        with open(fname, "w") as fp:
            json.dump(data_from_json, fp)
    else:
        data_from_json = {0: stats, "num entries": 1}
        with open(fname, "w") as fp:
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


def train_xent(net, trainloader, optimizer_obj, device):
    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup
    criterion = torch.nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    total = 1

    for inputs, targets in tqdm(trainloader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*targets.size(0)
        predicted = outputs.argmax(1)
        correct += torch.amin(predicted == targets, dim=[1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / total
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc


def train_xent_clipped(net, trainloader, optimizer_obj, device):
    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup
    clip = optimizer_obj.clip
    criterion = torch.nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    total = 1

    for inputs, targets in tqdm(trainloader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()*targets.size(0)
        predicted = outputs.argmax(1)
        correct += torch.amin(predicted == targets, dim=[1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / total
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc
