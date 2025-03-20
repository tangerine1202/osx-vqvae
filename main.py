import argparse
import utils
from pprint import pprint

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    timestamp = utils.readable_timestamp()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("-e-dim", "--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("-e", "--n_updates", type=int, default=5000)
    parser.add_argument("-in-dim", "--input_dim", type=int, default=1, help='1 for grayscale 3 for rgb')
    parser.add_argument("-ds", "--dataset", type=str, default='CIFAR10')
    parser.add_argument("-v", "--verbose", action="store_true")
    # whether or not to save model
    parser.add_argument("-o", "--output", type=str, default=timestamp)
    parser.add_argument("-viz", "--viz", action="store_true",
                        help="Whether to visualize the reconstructions")

    args = parser.parse_args()
    pprint(args.__dict__)

    return args


def main(args):
    print(f'Saving model and results to {args.output}.pth')

    # Load data and define batch data loaders
    training_data, validation_data, training_loader, validation_loader, x_train_var \
        = utils.load_data_and_data_loaders(args.dataset, args.batch_size)

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
        (x, _) = next(iter(training_loader))
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
            utils.save_model_and_results(
                model, results, hyperparameters, args.output)

            if args.viz:
                utils.save_reconstruction(x, x_hat, i, args.output)

            tqdm.write(f'Update #{i}, '
                       f'Recon Error: {np.mean(results["recon_errors"][-args.log_interval:]):.4f}, '
                       f'Loss: {np.mean(results["loss_vals"][-args.log_interval:]):.4f}, '
                       f'Perplexity: {np.mean(results["perplexities"][-args.log_interval:]):.4f}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
