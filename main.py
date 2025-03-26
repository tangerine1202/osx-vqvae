import argparse
import shutil
from pathlib import Path
from pprint import pprint

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    timestamp = utils.readable_timestamp()

    parser.add_argument("-e", "--n_updates", type=int, default=10000)
    parser.add_argument("-e-dim", "--embedding_dim", type=int, default=1)  # dim of each codebook item
    parser.add_argument("-in-dim", "--input_dim", type=int, default=1, help='1 for grayscale 3 for rgb')
    parser.add_argument("-img-size", "--img_size", nargs='+', type=int, default=[32, 32],
                        help='Resize input image to (height, width)')
    parser.add_argument("-ds", "--dataset", type=str, default='shape')
    parser.add_argument("-v", "--verbose", action="store_true")
    # whether or not to save model
    parser.add_argument("-o", "--output", type=str, default=timestamp, help="Output name (without path)")
    parser.add_argument("-nv", "--no_viz", action="store_true", help="Whether not to visualize the reconstructions")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100)

    args = parser.parse_args()
    args.img_size = args.img_size if len(args.img_size) == 2 else [args.img_size[0], args.img_size[0]]

    # Standardize output paths
    args.output_dir = Path('results') / args.output
    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.recon_dir = Path('results') / 'train_recon' / args.output
    args.recon_dir.mkdir(exist_ok=True, parents=True)

    pprint(args.__dict__)

    return args


def main(args):
    if args.recon_dir.exists():
        print(f'Warning: {args.recon_dir} already exists. Deleting...')
        shutil.rmtree(args.recon_dir)
    args.recon_dir.mkdir(exist_ok=True, parents=True)

    # Load data and define batch data loaders
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size, args.img_size)

    # Set up VQ-VAE model with components defined in ./models/ folder
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                  args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta,
                  in_dim=args.input_dim,
                  ).to(device)
    # Set up optimizer and training loop
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    model.train()

    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }

    for i in tqdm(range(args.n_updates)):
        (x, *_) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x, verbose=args.verbose)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            hyperparameters = args.__dict__
            model_path = utils.save_model_and_results(
                model, results, hyperparameters, args.output_dir)

            if not args.no_viz:
                utils.save_reconstruction(x, x_hat, i, args.recon_dir)

            tqdm.write(f'Update #{i}, '
                       f'Recon Error: {np.mean(results["recon_errors"][-args.log_interval:]):.4f}, '
                       f'Loss: {np.mean(results["loss_vals"][-args.log_interval:]):.4f}, '
                       f'Perplexity: {np.mean(results["perplexities"][-args.log_interval:]):.4f}')

    print(f'Model saved to {model_path}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
