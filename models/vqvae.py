
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False,
                 in_dim=3):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, output_dim=in_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False, 'Debugging mode: check shapes'

        return embedding_loss, x_hat, perplexity

    def encode(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        _, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            assert False
        return z_q


if __name__ == "__main__":
    # random data
    bs = 4
    x = np.random.random_sample((bs, 3, 160, 160))
    x = torch.tensor(x).float()

    # test encoder
    vqvae = VQVAE(
        h_dim=128, res_h_dim=64, n_res_layers=3,
        n_embeddings=512, embedding_dim=64, beta=0.25)

    emb_loss, x_hat, perplexity = vqvae(x)
    print('VQVAE out shape:', x_hat.shape)
    print('Embedding loss:', emb_loss)
    print('Perplexity:', perplexity)
