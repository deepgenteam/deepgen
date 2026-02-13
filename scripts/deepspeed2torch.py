import argparse
import torch
from xtuner.model.utils import guess_load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    model = guess_load_checkpoint(args.input)

    torch.save(model, args.output)
