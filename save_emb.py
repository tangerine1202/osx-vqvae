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
    parser.add_argument("-img-size", "--img_size", type=int, default=64)
    parser.add_argument("-ds", "--dataset", type=str, default='shape')
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Standardize output paths
    args.output_dir = Path(args.ckpt.parent)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    args.img_size = (args.img_size, args.img_size) if isinstance(args.img_size, int) else args.img_size
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
    _, _, training_loader, validation_loader, _ \
        = utils.load_data_and_data_loaders(args.dataset, args.batch_size, args.img_size)

    for name, loader in zip(['train', 'eval'], [training_loader, validation_loader]):
        fname_emb_dict = {}
        for x, label, filenames in loader:
            x = x.to(device)
            z_q = model.encode(x, args.verbose)
            z_q = z_q.cpu().detach().numpy()

            for i, fname in enumerate(filenames):
                f = str(Path(fname).with_suffix(''))
                fname_emb_dict[f] = z_q[i].flatten()

        print(f'----- {name} embeddings -----')
        emb_shape = list(fname_emb_dict.values())[0].shape
        print(f'# of embeddings: {len(fname_emb_dict)}')
        print(f'Embedding shape: {emb_shape}')
        print(f'Example Key: {list(fname_emb_dict.keys())[0]}')

        output_path = args.output_dir / f"{name}.npy"
        np.save(output_path, fname_emb_dict)
        print(f'Saved embeddings to {output_path}')

        print('Checking saved embeddings...')
        tmp_dict = np.load(output_path, allow_pickle=True).item()
        assert len(tmp_dict) == len(fname_emb_dict), "Number of embeddings mismatch"
        assert all(k in tmp_dict for k in fname_emb_dict.keys()), "Some keys are missing"
        assert all(k in fname_emb_dict for k in tmp_dict.keys()), "Some keys are missing"
        assert all(np.array_equal(v, tmp_dict[k]) for k, v in fname_emb_dict.items()), "Some values are different"
        print('Embeddings loaded successfully.')
        print('------------------------------')


if __name__ == "__main__":
    args = parse_args()
    main(args)
