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
    parser.add_argument("-nv", "--no_viz", action="store_true", help="Whether not to visualize the reconstructions")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Standardize output paths
    args.output_dir = Path(args.ckpt.parent)
    # args.output_dir.mkdir(exist_ok=True, parents=True)
    args.recon_dir = Path('results') / 'eval_recon' / args.output_dir.name
    args.recon_dir.mkdir(exist_ok=True, parents=True)

    args.img_size = (args.img_size, args.img_size) if isinstance(args.img_size, int) else args.img_size
    return args


def evaluate(args):
    # Load data and define batch data loaders
    _, validation_data, _, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size, args.img_size)

    print('Loading checkpoint from', args.ckpt)
    checkpoint = torch.load(args.ckpt, weights_only=False)
    hparams = SimpleNamespace(**checkpoint['hyperparameters'])
    state_dict = checkpoint['model']

    model = VQVAE(
        hparams.n_hiddens, hparams.n_residual_hiddens,
        hparams.n_residual_layers, hparams.n_embeddings, hparams.embedding_dim, hparams.beta,
        in_dim=hparams.input_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.verbose:
        print(model)

    model.eval()

    results = {
        'recon_errors': [],
        'embedding_losses': [],
        'perplexities': [],
    }

    # Create directory for evaluation results
    eval_dir = args.output_dir

    with torch.no_grad():
        for i, (x, *_) in enumerate(tqdm(validation_loader)):
            x = x.to(device)

            # Forward pass
            embedding_loss, x_hat, perplexity = model(x, verbose=False)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var

            # Save metrics
            results["recon_errors"].append(recon_loss.cpu().numpy())
            results["embedding_losses"].append(embedding_loss.cpu().numpy())
            results["perplexities"].append(perplexity.cpu().numpy())

            # Save sample reconstructions
            if i == 0 and not args.no_viz:
                utils.save_reconstruction(x, x_hat, 'evaluation', args.recon_dir)

    # Calculate average metrics
    avg_recon_error = np.mean(results["recon_errors"])
    avg_embedding_loss = np.mean(results["embedding_losses"])
    avg_perplexity = np.mean(results["perplexities"])

    # Print and save results
    print(f'Evaluation Results:')
    print(f'Avg Recon Error: {avg_recon_error:.4f}')
    print(f'Avg Embedding Loss: {avg_embedding_loss:.4f}')
    print(f'Avg Perplexity: {avg_perplexity:.4f}')

    # Save evaluation results to file
    with open(eval_dir / 'eval_results.txt', 'w') as f:
        f.write(f'Avg Recon Error: {avg_recon_error:.4f}\n')
        f.write(f'Avg Embedding Loss: {avg_embedding_loss:.4f}\n')
        f.write(f'Avg Perplexity: {avg_perplexity:.4f}\n')

    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
