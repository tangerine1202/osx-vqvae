import argparse
import utils
from pprint import pprint
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_viz(feat_dict, output_path=None, seed=42):
    def get_vertex_cnt(x): return int(x[4:].split('_')[0])

    def make_plot(data, output_path):
        # Visualize for point with different color
        plt.figure(figsize=(10, 10))
        for i, key in enumerate(keys):
            v_cnt = get_vertex_cnt(key)
            plt.scatter(data[i, 0], data[i, 1], c=f'C{v_cnt}', label=f'{v_cnt} vertices')
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        plt.legend(unique_labels.values(), unique_labels.keys())
        # Save the plot
        plt.savefig(output_path)
        print(f'Plot saved to {output_path}')

    keys = list(feat_dict.keys())
    features = np.stack(list(feat_dict.values()), axis=0)

    pca = PCA(n_components=2, random_state=seed)
    data_pca = pca.fit_transform(features)
    data_pca = (data_pca - data_pca.min()) / (data_pca.max() - data_pca.min())

    tsne = TSNE(n_components=2, random_state=seed)
    data_tsne = tsne.fit_transform(features)
    data_tsne = (data_tsne - data_tsne.min()) / (data_tsne.max() - data_tsne.min())

    make_plot(data_pca, output_path=f'{output_path}_pca.png')
    make_plot(data_tsne, output_path=f'{output_path}_tsne.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path)
    parser.add_argument("-img-size", "--img_size", type=int, default=32)
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

        plot_viz(fname_emb_dict, output_path=output_path.with_suffix(''))

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
