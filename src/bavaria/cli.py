"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mbavaria` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``bavaria.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``bavaria.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from bavaria import __version__
from bavaria.ensembles import EnsembleVAE
from bavaria.ensembles import BatchEnsembleVAE
from bavaria.data import load_data
from bavaria.data import load_batch_labels
from bavaria.data import one_hot_encode_batches
from bavaria.utils import resnet_vae_params
from bavaria.utils import resnet_vae_batch_params
from bavaria.utils import get_variable_regions
import scanpy as sc
import logging


def main(args=None):

    parser = argparse.ArgumentParser('bavaria',
                                     description=f'Negative multinomial variational auto-encoders - v{__version__}')

    parser.add_argument('-data', dest='data', type=str,
                        help="Data location. Matrix must be in mtx format or anndata .h5ad", required=True)
    parser.add_argument('-regions', dest='regions', type=str,
                        help="Regions location in bed format. Required if matrix is in mtx format.")
    parser.add_argument('-barcodes', dest='barcodes', type=str,
                        help="Barcode file in tsv format (with header). First column denotes the barcode. "
                             "Required if matrix is in mtx format.",
                        )

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
                        default=0.0,
                        help="Dropout rate applied at the inital layer (e.g. input accessibility profile). Default=0.0")
    parser.add_argument("-hidden_e_dropout", dest="hidden_e_dropout", type=float,
                        default=0.0,
                        help="Dropout applied in each residual block of the encoder. Default=0.0")
    parser.add_argument("-hidden_d_dropout", dest="hidden_d_dropout", type=float,
                        default=0.0,
                        help="Dropout applied after each decoder hidden layer. Default=0.0")
    parser.add_argument("-feature_fraction", dest="feature_fraction", type=float, default=1.,
                        help="Whether to use a random subset of features. feature_fraction determines the proportion of features to use. Default=1.")
    parser.add_argument("-batches", dest="batches", type=str, default=None,
                        help="Table in tsv format defining the cell batches. "
                             "The first columns should represent the barcode "
                             "while the remaining columns represent the batches as categorical labels.")
             
    parser.add_argument("-batchnames", dest="batchnames", type=str, nargs='+', default=[],
                        help="Batch names in the anndata dataset. ")
    parser.add_argument("-modelname", dest="modelname", type=str, default='vae', choices=[
                                                                                          'bavaria-0', 'bavaria', 
                                                                                          'bcvae', 'bcvae2', 
                                                                                           'vae',
                                                                                           'vae-ml',
                                                                                         ],
                        help="Model name for batch correction. Default: vae")
    parser.add_argument('-resolution', dest='resolution', type=float, default=1.,
                        help="Resolution for Louvain clustering analysis.")


             
    parser.add_argument('-nhidden_b', dest='nhidden_b', type=int,
                        default=32,
                        help="Number of hidden neurons for batch predictor. "
                             "Default: 32.")

    args = parser.parse_args()

    matrix = args.data
    regions = args.regions
    barcodes = args.barcodes
    batches = args.batches
    
    batchnames=args.batchnames
    # load the dataset
    adata = load_data(matrix, regions, barcodes)

    params = resnet_vae_params(args)

    if len(batchnames)>0:
        print('using batch correction from adata')
        adata = one_hot_encode_batches(adata, batchnames)
        params.update(resnet_vae_batch_params(adata, batchnames))

        metamodel = BatchEnsembleVAE(args.modelname, params,
                                args.nrepeat, args.output,
                                args.overwrite,
                                args.feature_fraction,
                                params['batchnames'])
    else:
        metamodel = EnsembleVAE(params,
                            args.nrepeat, args.output,
                            args.overwrite,
                            name=args.modelname,
                            feature_fraction=args.feature_fraction)

    metamodel.fit(adata, epochs=args.epochs, batch_size=args.batch_size)

    adata = metamodel.encode(adata)

    sc.pp.neighbors(adata, n_neighbors=15, use_rep="bavaria-ensemble")
    sc.tl.louvain(adata, resolution=args.resolution)
    sc.tl.umap(adata)

    
    if 'batchnames' not in params:
        params['batchnames'] = None
    adata = get_variable_regions(adata, batches=params['batchnames'])
    adata.write(os.path.join(args.output, "analysis.h5ad"))
    print('saved to ' + os.path.join(args.output, "analysis.h5ad"))
