""" utils.py
    utility functions and classes
    Developed as part of DeepThinking2 project
    April 2021
"""

import datetime
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms as transforms
from icecream import ic
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
from icecream import ic

from models.recur_resnet_segment import recur_resnet
from models.resnet_segment import ff_resnet


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def get_dataloaders(train_batch_size, test_batch_size, data_path, shuffle=True):

    train_inputs_np = np.load(os.path.join(data_path, "train_small/inputs.npy"))
    train_targets_np = np.load(os.path.join(data_path, "train_small/solutions.npy"))
    test_inputs_np = np.load(os.path.join(data_path, "test_large/inputs.npy"))
    test_targets_np = np.load(os.path.join(data_path, "test_large/solutions.npy"))

    train_inputs = torch.from_numpy(train_inputs_np).float().permute(0, 3, 1, 2)
    train_targets = torch.from_numpy(train_targets_np).permute(0, 3, 1, 2)
    test_inputs = torch.from_numpy(test_inputs_np).float().permute(0, 3, 1, 2)
    test_targets = torch.from_numpy(test_targets_np).permute(0, 3, 1, 2)

    train_data = MazeDataset(train_inputs, train_targets, 5)
    test_data = MazeDataset(test_inputs, test_targets, 7)

    trainloader = data.DataLoader(train_data, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(test_data, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    return trainloader, testloader


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

    if "recur" in model:
        base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        iters = net.iters
    else:
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
    net.load_state_dict(state_dict["net"])
    net = net.to(device)
    return net, state_dict["epoch"], state_dict["optimizer"]


class MazeDataset(data.Dataset):
    """This is a dataset class for mazes.
    padding and cropping is done correctly within this class for
    small, medium, and large mazes.
    """
    def __init__(self, inputs, targets, maze_size):
        self.inputs = inputs
        self.targets = targets
        self.padding = {5: 4, 6: 2, 7: 0}[maze_size]
        self.pad = transforms.Pad(self.padding)

    def __getitem__(self, index):
        x = self.pad(self.inputs[index])
        y = self.pad(self.targets[index])
        i = random.randint(0, 2*self.padding)
        j = random.randint(0, 2*self.padding)

        return x[:, i:i+32, j:j+32], y[:, i:i+32, j:j+32]

    def __len__(self):
        return self.inputs.size(0)


def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


@dataclass
class OptimizerWithSched:
    """Attributes for optimizer, lr schedule, and lr warmup"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"


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
            inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
            outputs = net(inputs)

            predicted = outputs.argmax(1) * inputs.max(1)[0]
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def test_max_conf(net, testloader, device):

    net.eval()
    net.to(device)
    correct = 0
    confidence = torch.zeros(net.iters)
    total = 0
    total_pixels = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
            net(inputs)
            confidence_array = torch.zeros(net.iters, inputs.size(0))
            for i, thought in enumerate(net.thoughts):
                conf = torch.nn.functional.softmax(thought.detach(), dim=1).max(1)[0] \
                       * inputs.max(1)[0]
                confidence[i] += conf.sum().item()
                confidence_array[i] = conf.sum([1, 2]) / inputs.max(1)[0].sum([1, 2])

            exit_iter = confidence_array.argmax(0)

            best_thoughts = net.thoughts[exit_iter, torch.arange(net.thoughts.size(1))].squeeze()
            if best_thoughts.shape[0] != inputs.shape[0]:
                best_thoughts = best_thoughts.unsqueeze(0)
            predicted = best_thoughts.argmax(1) * inputs.max(1)[0]
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()

            total_pixels += inputs.max(1)[0].sum().item()
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

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0
    total_pixels = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
        optimizer.zero_grad()
        outputs = net(inputs)

        n, c, h, w = outputs.size()
        reshaped_outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_outputs = reshaped_outputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        reshaped_outputs = reshaped_outputs.view(-1, c)

        reshaped_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_inputs = reshaped_inputs.mean(3).unsqueeze(-1)
        reshaped_inputs = reshaped_inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, 1) >= 0]
        reshaped_inputs = reshaped_inputs.view(-1, 1)
        path_mask = (reshaped_inputs > 0).squeeze()

        mask = targets >= 0.0
        reshaped_targets = targets[mask]

        loss = criterion(reshaped_outputs, reshaped_targets)
        loss = loss[path_mask].mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * path_mask.size(0)
        total_pixels += path_mask.size(0)

        predicted = outputs.argmax(1) * inputs.max(1)[0]
        correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / total_pixels
    acc = 100.0 * correct / total
    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc