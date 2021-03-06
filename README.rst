========
BAVARIA
========

BAVARIA is python package that implements a
Batch-adversarial Variational auto-encoder with Negative Multinomial reconstruction loss for single-cell ATAC-seq analysis.

.. image:: bavaria_scheme.svg
  :width: 600

In particular, the model can be used to extract a latent feature representation of
a cell which can be used for downstream analysis tasks, including cell cluster,
cell identification, etc.
The package is freely available under a GNU Lesser General Public License v3 or later (LGPLv3+)

Installation
============

You can install the package version v0.1.0 via

::

    pip install https://github.com/BIMSBbioinfo/bavaria/archive/v0.1.0.zip

Alternatively, you can install the latest version from the master branch using

::

    pip install git+https://github.com/BIMSBbioinfo/bavaria.git

Documentation
=============

BAVARIA offers a command line interface that fits an ensemble of BAVARIA models
given a raw count matrix (-data)
Subsequently, the model parameters and latent features
are stored in the output directory (-output)

::

   bavaria -data adata.h5ad \
         -output <outputdir> \
         -epochs 200 \
         -nrepeats 10 \
         -nlatent 15 \
         -batchnames batch \
         -modelname bavaria
 
Additional information on available hyper-parameters are available through

::

  bavaria -h

Tutorial
========


Below you find links to the tutorials. 
The tutorials will require jupyter and other resources which are defined in
:code:`tutorial/requirements.txt`. Using the requirements file you instantiate
a new conda environment using 

.. code:: bash

    conda create --name bavaria_tutorial --file tutorial/requirements.txt


+----------------------------------------------------+
| Example notebooks                                  |
+====================================================+
| `Data preparation PBMC integration`_               |
+----------------------------------------------------+
| `Using BAVARIA to integrate PBMC data`_            |
+----------------------------------------------------+

.. _`Data preparation PBMC integration`: https://nbviewer.jupyter.org/github/BIMSBbioinfo/bavaria/blob/master/tutorial/00_preparation.ipynb
.. _`Using BAVARIA to integrate PBMC data`: https://nbviewer.jupyter.org/github/BIMSBbioinfo/bavaria/blob/master/tutorial/01_pbmc_integration.ipynb

