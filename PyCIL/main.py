import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--moles', action='store_true',
                        help='if model will be poisoned by moles')
    parser.add_argument('--rho', default=0.5, type=float,
                        help='provide threshold rho value')
    parser.add_argument('--test', action='store_true',
                        help='to write logs to test file')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')


    return parser


if __name__ == '__main__':
    main()
