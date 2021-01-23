import os
from collections import OrderedDict
import sys
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from keras.models import load_model

def resnet_vae_batch_params(adata, batchnames):
    columns = batchnames
    ncats = [adata.obs[c].unique().shape[0] for c in columns]
    nullprobs = [adata.obsm[c].mean(0) for c in columns]
    params = OrderedDict([
       ('batchnames', columns),
       ('nbatchcats', ncats),
       ('batchnullprobs', nullprobs),
         ])
    return params


def resnet_vae_params(args):
    params = OrderedDict(
      [
        ('nlayers_d', args.nlayers_d),
        ('nhidden_e', args.nhidden_e),
        ('nlayers_e', args.nlayers_e),
        ('nsamples', args.nsamples),
        ('nhiddendecoder', args.nhidden_d),
        ('inputdropout', args.inputdropout),
        ('hidden_e_dropout', args.hidden_e_dropout),
        ('hidden_d_dropout', args.hidden_d_dropout),
        ('nhiddenbatcher', args.nhidden_b),
        ('nlayersbatcher', 2),
        ('nlasthiddenbatcher', 5),
        ('latentdims', args.nlatent),
      ]
    )

    return params



class MultiLogisticRegression:
    def __init__(self, adata, cluster, batches):
        self.cluster = cluster
        self.batches = batches
        nregions = adata.shape[-1]
        nclusters = adata.obsm[cluster].shape[-1]
        nbatches = sum([adata.obsm[b].shape[-1] for b in batches])
        self.optimizer = tf.keras.optimizers.Adam()
        init = tf.random_normal_initializer(stddev=.001)
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.W = tf.Variable(init(shape=(nregions,nclusters)))
        self.B = tf.Variable(init(shape=(nregions,nbatches)))
        self.R = tf.Variable( init(shape=(nregions,)))
        self.a = tf.Variable( init(shape=(nregions,1)))

    @property
    def params(self):
        return [self.W,self.B,self.R,self.a]

    def loss(self, y, pred):
        y_pos, y_neg = y
        return - tf.math.reduce_mean(tf.math.xlogy(y_pos, pred)) - \
               tf.math.reduce_mean(tf.math.xlogy(y_neg, 1.-pred)) + \
               1e-4*tf.math.reduce_mean(tf.math.square(self.W))

    def predict(self, inputs):
        xc, xb, xr = inputs

        l1 = tf.einsum('rk,mk->rm',self.W, xc)
        l2 = tf.einsum('rb,mb->rm', self.B, xb)
        l3 = tf.einsum('r,m->rm', self.R, xr)

        return tf.math.sigmoid(l1 + l2 + l3 + self.a)

    def update(self, inputs, y):

        with tf.GradientTape(persistent=True) as tape:
            tot_loss = self.loss(y, self.predict(inputs))

        g = tape.gradient(tot_loss, self.params)
        self.optimizer.apply_gradients(zip(g, self.params))

        return tot_loss

    def fit(self, adata, epochs=10):

        clusters = adata.obsm[self.cluster]
        nbatches = len(self.batches)
        batches = np.concatenate([adata.obsm[batch] for batch in self.batches], axis=1)
        
        #c x m
        M = np.concatenate([clusters, batches],axis=1)

        # m x m
        Mmap = pd.DataFrame(M).drop_duplicates().values
        M = M.dot(Mmap.T)
        # c x m  cell to cluster/batch combination
        M[M<(1+nbatches)] = 0
        M[M>=(1+nbatches)] = 1

        # region by cells x cells by M
        y_pos = adata.X.T.dot(M)
        y_neg = np.ones(adata.shape[0]).dot(M) - y_pos

        # m x b
        M2kmap = Mmap[:,:clusters.shape[1]]
        M2bmap = Mmap[:,clusters.shape[1]:]

        # m x k
        readdepth = adata.obs['rdepth'].values.astype('float32')/adata.shape[-1]
        readdepth = M.T.dot(readdepth)

        X = (M2kmap, M2bmap, readdepth)
        y = (y_pos,y_neg)
        for i in range(epochs):
            loss = self.update(X, y)


def get_variable_regions(adata, groupby="louvain",
                         batches=['batch']):
    """Extract variable regions relative to clustering."""

    if 'readdepth' not in adata.obs.columns:
        adata.obs['rdepth'] = np.asarray(adata.X.sum(1)).flatten()

    if batches is None:
        batches = ['dummybatch']
        adata.obs['dummybatch'] = 'dummy'
    for batch in batches or []:
        if batch not in adata.obsm:
            adata.obsm[batch] = OneHotEncoder(sparse=False).fit_transform(np.asarray(adata.obs[batch].values.tolist()).reshape(-1,1))
    if groupby not in adata.obsm:
        adata.obsm[groupby] = OneHotEncoder(sparse=False).fit_transform(np.asarray(adata.obs[groupby].values.tolist()).reshape(-1,1))

    mlr = MultiLogisticRegression(adata, groupby, batches)
    mlr.fit(adata, epochs=500)
    scores = mlr.W.numpy()

    np.testing.assert_equal(scores.shape, (adata.shape[1], adata.obsm[groupby].shape[1]))

    adata.varm['cluster_spec_region'] = scores

    return adata
