"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mnmvae` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``nmvae.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``nmvae.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from nmvae import VAE
from nmvae.resnet import MetaVAE
from nmvae.resnet import load_data
from nmvae.resnet import resnet_vae_params
#from scregseg.countmatrix import CountMatrix
import logging


def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', dest='data', type=str,
                        help="Matrix location. Matrix must be in mtx format", required=True)
    parser.add_argument('-regions', dest='regions', type=str,
                        help="Regions location in bed format.", required=True)
    parser.add_argument('-barcodes', dest='barcodes', type=str,
                        help="Barcode file in tsv format (with header). First column denotes the barcode.",
                        required=True)

    parser.add_argument('-output', dest='output', type=str,
                        help="Output directory", required=True)


    parser.add_argument('-nlatent', dest='nlatent', type=int, default=10,
                        help="Latent dimensions. Default: 10")

    parser.add_argument('-epochs', dest='epochs', type=int, default=100,
                        help="Number of epochs. Default: 100.")
    parser.add_argument('-nrepeat', dest='nrepeat', type=int, default=10,
                        help="Number of repeatedly fitted models. "
                             "Default: 10.")
    parser.add_argument('-batch_size', dest='batch_size', type=int,
                        default=128,
                        help='Batch size. Default: 128.')
    parser.add_argument('-overwrite', dest='overwrite',
                        action='store_true', default=False)

    parser.add_argument('-nlayers_d', dest='nlayers_d', type=int,
                        default=1,
                        help="Number of decoder hidden layers. Default: 1.")
    parser.add_argument('-nhidden_e', dest='nhidden_e', type=int,
                        default=512,
                        help="Number of neurons per encoder layer. "
                             "This number is constant across the encoder. Default: 512.")
    parser.add_argument('-nlayers_e', dest='nlayers_e', type=int,
                        default=20,
                        help="Number of residual blocks for the encoder. Default: 20.")
    parser.add_argument('-nsamples', dest='nsamples', type=int,
                        default=2,
                        help="Number of latent samples drawn per datapoints. "
                             "Usually in most VAE applications, one sample is draw, "
                             "However, for very sparse and noisy ATAC-seq data drawing multiple samples"
                             " may decrease the noisy estimation of the gradient during training, although"
                             " the effect is usually relatively small. Default: 2.")
    parser.add_argument('-nhidden_d', dest='nhidden_d', type=int,
                        default=16,
                        help="Number of neurons per decoder hidden layer. "
                             "Usually it is important to set nhidden_d as well as nlatent to relatively small values. "
                             " Too high numbers for these parameters degrades the quality of the latent features. "
                             "Default: 16.")
    parser.add_argument('-inputdropout', dest='inputdropout', type=float,
                        default=0.15,
                        help="Dropout rate applied at the inital layer (e.g. input accessibility profile). Default=0.15")
    parser.add_argument("-hidden_e_dropout", dest="hidden_e_dropout", type=float,
                        default=0.3,
                        help="Dropout applied in each residual block of the encoder. Default=0.3")
    parser.add_argument("-hidden_d_dropout", dest="hidden_d_dropout", type=float,
                        default=0.3,
                        help="Dropout applied after each decoder hidden layer. Default=0.3")
             

    args = parser.parse_args()
    print(args)

    matrix = args.data
    regions = args.regions# if args.regions is not None else matrix + '.bed'
    barcodes = args.barcodes# if args.barcodes is not None else matrix + '.bct'

    # load the dataset
    cmat, barcode, regions = load_data(matrix, regions, barcodes)

    input_shape = cmat.shape[-1]
    params = resnet_vae_params(input_shape, args)
    metamodel = MetaVAE(params,
                        args.nrepeat, args.output,
                        args.overwrite)

    metamodel.fit(cmat, epochs=args.epochs, batch_size=args.batch_size)

    metamodel.encode(cmat, barcode)
