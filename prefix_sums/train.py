"""train.py
   Train, test, and save models
   Developed as part of Easy-To-Hard project
   April 2021
"""

import argparse
import os
import sys
from collections import OrderedDict

from icecream import ic
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import warmup
from utils import train, test, OptimizerWithSched, load_model_from_checkpoint, \
    get_dataloaders, to_json, get_optimizer, to_log_file, now, get_model


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def main():

    print("\n_________________________________________________\n")
    print(now(), "train.py main() running.")

    parser = argparse.ArgumentParser(description="Deep Thinking")
    parser.add_argument("--checkpoint", default="check_default", type=str,
                        help="where to save the network")
    parser.add_argument("--clip", default=1.0, help="max gradient magnitude for training")
    parser.add_argument("--data_path", default="../data", type=str, help="path to data files")
    parser.add_argument("--debug", action="store_true", help="debug?")
    parser.add_argument("--depth", default=8, type=int, help="depth of the network")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs for training")
    parser.add_argument("--eval_data", default=20, type=int, help="what size eval data")
    parser.add_argument("--json_name", default="test_stats", type=str, help="name of the json file")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default="step", type=str, help="which kind of lr decay")
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")
    parser.add_argument("--lr_schedule", nargs="+", default=[100, 150], type=int,
                        help="how often to decrease lr")
    parser.add_argument("--model", default="conv_net", type=str, help="model for training")
    parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle",
                        help="shuffle training data?")
    parser.add_argument("--optimizer", default="sgd", type=str, help="optimizer")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--save_json", action="store_true", help="save json")
    parser.add_argument("--save_period", default=None, type=int, help="how often to save")
    parser.add_argument("--test_batch_size", default=500, type=int, help="batch size for testing")
    parser.add_argument("--test_iterations", default=None, type=int,
                        help="how many, if testing with a different number iterations")
    parser.add_argument("--test_mode", default="default", type=str, help="testing mode")
    parser.add_argument("--train_batch_size", default=128, type=int, help="batch size for training")
    parser.add_argument("--train_data", default=16, type=int, help="what size train data")
    parser.add_argument("--train_log", default="train_log.txt", type=str,
                        help="name of the log file")
    parser.add_argument("--train_mode", default="xent", type=str, help="training mode")
    parser.add_argument("--train_split", default=0.8, type=float,
                        help="percentile of difficulty to train on")
    parser.add_argument("--val_period", default=20, type=int, help="how often to validate")
    parser.add_argument("--warmup_period", default=5, type=int, help="warmup period")
    parser.add_argument("--width", default=4, type=int, help="width of the network")

    args = parser.parse_args()
    args.train_mode, args.test_mode = args.train_mode.lower(), args.test_mode.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_period is None:
        args.save_period = args.epochs

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # TensorBoard
    train_log = args.train_log
    try:
        array_task_id = train_log[:-4].split("_")[-1]
    except:
        array_task_id = 1
    if not args.debug:
        to_log_file(args, args.output, train_log)
        writer = SummaryWriter(log_dir=f"{args.output}/runs/{train_log[:-4]}")
    else:
        writer = SummaryWriter(log_dir=f"{args.output}/debug/{train_log[:-4]}")
    ####################################################
    #               Dataset and Network and Optimizer
    trainloader, testloader, evalloader = get_dataloaders(args.train_batch_size, args.test_batch_size, args.train_data,
                                                          args.eval_data, args.train_split, shuffle=args.shuffle)

    # load model from path if a path is provided
    if args.model_path is not None:
        print(f"Loading model from checkpoint {args.model_path}...")
        net, start_epoch, optimizer_state_dict = load_model_from_checkpoint(args.model,
                                                                            args.model_path,
                                                                            args.width,
                                                                            args.depth)
        start_epoch += 1

    else:
        net = get_model(args.model, args.width, args.depth)
        start_epoch = 0
        optimizer_state_dict = None

    device_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if args.test_iterations and len(device_ids) > 1:
        print(f"{ic.format()}: Can't test on multiple GPUs. Exiting")
        sys.exit()
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net = net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    optimizer = get_optimizer(args.optimizer, args.model, net, args.lr)

    if args.debug:
        print(net)
    print(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    print(f"Training will start at epoch {start_epoch}.")

    if optimizer_state_dict is not None:
        print(f"Loading optimizer from checkpoint {args.model_path}...")
        optimizer.load_state_dict(optimizer_state_dict)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=args.warmup_period)

    if args.lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=args.lr_schedule, gamma=args.lr_factor,
                                   last_epoch=-1)
    elif args.lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1,
                                         verbose=False)
    else:
        print(f"{ic.format()}: Learning rate decay style {args.lr_decay} not yet implemented."
              f"Exiting.")
        sys.exit()

    optimizer_obj = OptimizerWithSched(optimizer, lr_scheduler, warmup_scheduler, args.clip)
    torch.backends.cudnn.benchmark = True
    ####################################################

    ####################################################
    #        Train
    print(f"==> Starting training for {args.epochs - start_epoch} epochs...")

    for epoch in range(start_epoch, args.epochs):

        loss, acc = train(net, trainloader, args.train_mode, optimizer_obj, device)

        print(f"{now()} Training loss at epoch {epoch}: {loss}")
        print(f"{now()} Training accuracy at epoch {epoch}: {acc}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            print(f"{ic.format()} Loss is nan, exiting...")
            sys.exit()

        # TensorBoard loss writing
        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}", optimizer.param_groups[i]["lr"], epoch)

        if (epoch + 1) % args.val_period == 0:
            train_acc = test(net, trainloader, args.test_mode, device)
            test_acc = test(net, testloader, args.test_mode, device)
            eval_acc = test(net, evalloader, args.test_mode, device)
            # eval_acc = 0
            print(f"{now()} Training accuracy: {train_acc}")
            print(f"{now()} Testing accuracy: {test_acc}")
            print(f"{now()} Eval accuracy (hard data): {eval_acc}")

            stats = [train_acc, test_acc, eval_acc]
            stat_names = ["train_acc", "test_acc", "eval_acc"]
            for stat_idx, stat in enumerate(stats):
                stat_name = os.path.join("val", stat_names[stat_idx])
                writer.add_scalar(stat_name, stat, epoch)

        if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs:
            state = {
                "net": net.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict()
            }
            out_str = os.path.join(args.checkpoint,
                                   f"{args.model}_{args.optimizer}"
                                   f"_depth={args.depth}"
                                   f"_width={args.width}"
                                   f"_lr={args.lr}"
                                   f"_batchsize={args.train_batch_size}"
                                   f"_epoch={args.epochs-1}"
                                   f"_{array_task_id}.pth")

            print(f"{now()} Saving model to: ", args.checkpoint, " out_str: ", out_str)
            if not os.path.isdir(args.checkpoint):
                os.makedirs(args.checkpoint)
            torch.save(state, out_str)

    writer.flush()
    writer.close()
    ####################################################

    ####################################################
    #        Test
    print("==> Starting testing...")

    if args.test_iterations > 0:
        assert isinstance(net.module.iters, int), f"{ic.format()} Cannot test " \
                                                  f"feed-forward model with iterations."
        net.module.iters = args.test_iterations

    test_acc = test(net, testloader, args.test_mode, device)
    train_acc = test(net, trainloader, args.test_mode, device)
    eval_acc = test(net, evalloader, args.test_mode, device)
    # eval_acc = 0

    print(f"{now()} Training accuracy: {train_acc}")
    print(f"{now()} Testing accuracy: {test_acc}")
    print(f"{now()} Eval accuracy (hard data): {eval_acc}")

    model_name_str = f"{args.model}_depth={args.depth}_width={args.width}"
    stats = OrderedDict([("epochs", args.epochs),
                         ("eval_acc", eval_acc),
                         ("learning rate", args.lr),
                         ("lr", args.lr),
                         ("lr_factor", args.lr_factor),
                         ("model", model_name_str),
                         ("num_params", pytorch_total_params),
                         ("optimizer", args.optimizer),
                         ("test_acc", test_acc),
                         ("test_iter", args.test_iterations),
                         ("test_mode", args.test_mode),
                         ("train_acc", train_acc),
                         ("train_batch_size", args.train_batch_size),
                         ("train_mode", args.train_mode)])

    if args.save_json:
        args.json_name += ".json"
        to_json(stats, args.output, args.json_name)
    ####################################################


if __name__ == "__main__":
    main()
