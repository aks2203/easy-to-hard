import argparse
import os

from utils import now


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def main():

    print("\n_________________________________________________\n")
    print(now(), "train.py main() running.")

    parser = argparse.ArgumentParser(description="Deep Thinking")
    parser.add_argument("--depth", default=8, type=int, help="depth of the network")
    parser.add_argument("--eval_data", default=20, type=int, nargs="+", help="what size eval data")
    parser.add_argument("--json_name", default="test_stats", type=str, help="name of the json file")
    parser.add_argument("--model", default="conv_net", type=str, help="model for training")
    parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--test_batch_size", default=500, type=int, help="batch size for testing")
    parser.add_argument("--test_iterations", default=None, type=int, nargs="+",
                        help="how many, if testing with a different number iterations")
    parser.add_argument("--test_mode", default="default", type=str, help="testing mode")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="batch size for training")
    parser.add_argument("--train_data", default=20, type=int, help="what size training data")
    parser.add_argument("--width", default=4, type=int, help="width of the network")

    args = parser.parse_args()

    for eval_data in args.eval_data:
        for test_iterations in args.test_iterations:
            print(f"Eval data: {eval_data}, Test iterations: {test_iterations}")
            cmd_str = f"python train.py " \
                      f"--depth {args.depth} " \
                      f"--eval_data {eval_data} " \
                      f"--model {args.model} " \
                      f"--model_path {args.model_path} " \
                      f"--output {args.output}_test_{eval_data} " \
                      f"--test_batch_size {args.test_batch_size} " \
                      f"--test_iterations {test_iterations} " \
                      f"--test_mode {args.test_mode} " \
                      f"--train_batch_size {args.train_batch_size} " \
                      f"--train_data {args.train_data} " \
                      f"--width {args.width} " \
                      f"--epochs 0 " \
                      f"--save_json " \
                      f"--json_name {args.json_name}"
            os.system(f"{cmd_str}")


if __name__ == "__main__":
    main()
