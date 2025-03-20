from pathlib import Path
import time
import os

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.block import BlockDataset, LatentBlockDataset
from datasets.custom import ShapeDataset


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_shape_dataset():
    TRAIN_DIR = Path(__file__).parent / 'train-vqvae' / 'clean_pegs'
    VAL_DIR = Path(__file__).parent / 'val-vqvae' / 'clean_pegs'

    IMG_SIZE = (64, 64)

    train = ShapeDataset(
        dir_path=TRAIN_DIR,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize(IMG_SIZE),
        ]))
    val = ShapeDataset(
        dir_path=TRAIN_DIR,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize(IMG_SIZE),
        ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val


def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True, transform=None)

    val = LatentBlockDataset(data_file_path, train=False, transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'shape':
        training_data, validation_data = load_shape_dataset()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    else:
        raise ValueError(f'Invalid dataset: {dataset}')

    print(f"Variance of dataset: {x_train_var}")

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, filename):
    SAVE_MODEL_PATH = Path(os.getcwd() + '/results')
    SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, str(SAVE_MODEL_PATH) + '/vqvae_data_' + filename + '.pth')


def save_reconstruction(x, x_hat, i, filename):
    # Save the reconstruction images
    save_dir_path = Path(os.getcwd()) / 'results' / f'reconstructions_{filename}'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    bs, c, h, w = x.shape
    bs_sqrt = int(np.sqrt(bs))

    fig, axes = plt.subplots(bs_sqrt, bs_sqrt, figsize=(10, 10))
    for ax, img, rec in zip(axes.flatten(), x, x_hat):
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        rec = rec.permute(1, 2, 0).detach().cpu().numpy()
        diff = np.abs(img - rec)
        ax.imshow(diff)
        ax.axis('off')
    fig.suptitle(f'Absolute Difference {i}')
    plt.tight_layout()
    plt.savefig(save_dir_path / f'abs_diff_{i}.png')
    plt.close(fig)

    n_pairs = 2
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 10))
    for idx, [img, rec] in enumerate(zip(x[:n_pairs], x_hat[:n_pairs])):
        ax_ori, ax_rec = axes[idx]
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        rec = rec.permute(1, 2, 0).detach().cpu().numpy()
        ax_ori.imshow(img)
        ax_rec.imshow(rec)
        ax_ori.set_title(f'Original {idx}')
        ax_rec.set_title(f'Reconstructed {idx}')
        ax_ori.axis('off')
        ax_rec.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir_path / f'reconstruction_{i}.png')
    plt.close(fig)
