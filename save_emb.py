import argparse
import utils
from pprint import pprint
from pathlib import Path
from types import SimpleNamespace

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path)
    parser.add_argument("-ds", "--dataset", type=str, default='shape')
    parser.add_argument("-o", "--output", type=Path, help="Output embedding file, default to ckpt name", default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.output = args.output or args.ckpt.with_suffix('.npy')
    return args


def main(args):

    print('Loading checkpoint from', args.ckpt)
    checkpoint = torch.load(args.ckpt, weights_only=False)
    hparams = SimpleNamespace(**checkpoint['hyperparameters'])
    state_dict = checkpoint['model']

    if args.verbose:
        print('----- Model hyperparameters -----')
        pprint(hparams)
        print('---------------------------------')

    model = VQVAE(
        hparams.n_hiddens, hparams.n_residual_hiddens,
        hparams.n_residual_layers, hparams.n_embeddings, hparams.embedding_dim, hparams.beta,
        in_dim=hparams.input_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.verbose:
        print(model)

    # Load data and define batch data loaders
    training_data, validation_data, training_loader, validation_loader, x_train_var \
        = utils.load_data_and_data_loaders(args.dataset, args.batch_size)

    filename_emb_dict = {}
    for x, filenames in training_loader:
        x = x.to(device)
        z_q = model.encode(x, args.verbose)
        z_q = z_q.cpu().detach().numpy()
        filename_emb_dict.update({f: z_q[i].flatten() for i, f in enumerate(filenames)})

    print(f'Number of embeddings: {len(filename_emb_dict)}')
    print(f'Example embedding: {list(filename_emb_dict.values())[0].shape}')

    np.save(args.output, filename_emb_dict)
    print(f'Saved embeddings to {args.output}')

    print('Checking saved embeddings...')
    tmp_dict = np.load(args.output, allow_pickle=True).item()
    assert len(tmp_dict) == len(filename_emb_dict), "Number of embeddings mismatch"
    assert all(k in tmp_dict for k in filename_emb_dict.keys()), "Some keys are missing"
    assert all(k in filename_emb_dict for k in tmp_dict.keys()), "Some keys are missing"
    assert all(np.array_equal(v, tmp_dict[k]) for k, v in filename_emb_dict.items()), "Some values are different"
    print('Embeddings loaded successfully.')


if __name__ == "__main__":
    args = parse_args()
    main(args)
