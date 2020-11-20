========
NMVAE
========

NMVAE is python package that implements a
Negative Multinomial Variational Auto-Encoder for single-cell ATAC-seq analysis.
In particular, the model can be used to extract a latent feature representation of
a cell which can be used for downstream analysis tasks, including cell cluster,
cell identification, etc.
The package is freely available under a GNU Lesser General Public License v3 or later (LGPLv3+)

Installation
============

::

    pip install https://github.com/BIMSBbioinfo/nmvae/archive/v0.0.1.zip


Documentation
=============

The tool offers a command line interface that, given a count matrix, fits a models and predicts
the corresponding latent features and stores them in the output directory.
The minimally required options are

::

   nmvae -data <matrix.mtx> 
         -regions <regions.bed> 
         -barcodes <barcodes.tsv>
         -output <outputdir>
 
matrix.mtx represents a regions by cells matrix in matrix market format.
regions.bed and barcodes.tsv represent the row and column annotations.
outputdir represents the output directory, in which the latent features are stored (latent.tsv) as well as the trained models.

Additional hyperparameters for the networks, such as batch-sizes, number of epochs, etc. are
initialized with sensible default parameters. 
However, they might need to be adjusted
depending on the dataset at hand.
In particular, we often found 
the number of latent features (nlatent), number of epochs (epochs), batch-size (batch_size)
to require adjustments depending on the dataset.
