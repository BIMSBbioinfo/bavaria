import os
from itertools import product
from collections import OrderedDict
import sys
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
        nbatches = np.prod([adata.obsm[b].shape[-1] for b in batches])
        nblocks = nclusters * np.prod([adata.obsm[b].shape[-1] for b in batches])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.1)
        init = tf.random_normal_initializer(stddev=.001)
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.W = tf.Variable(init(shape=(nregions,nclusters)))
        self.B = tf.Variable(init(shape=(nregions,nbatches)))
        self.a = tf.Variable(init(shape=(nregions,1)))
        self.b = tf.Variable(init(shape=(1, nblocks)))

    @property
    def params(self):
        return [self.W,self.B,self.a,self.b]

    def loss(self, adata):
        X, y = self._prepare_data(adata)
        return self._loss(y, self._predict(X))

    def _loss(self, y, pred):
        y_pos, y_neg = y
        return - tf.math.reduce_mean(tf.math.xlogy(y_pos, pred)) - \
               tf.math.reduce_mean(tf.math.xlogy(y_neg, 1.-pred)) + \
               1e-4*tf.math.reduce_mean(tf.math.square(self.W))

    def _predict(self, inputs):
        # M cells x cbcomb
        # mmap cbcomb x [cluster+batches]'
        mmap = inputs
        par = tf.concat([self.W, self.B],axis=1)
        #par  'regions x [cluster+batches]'
        l1 = tf.einsum('rp,mp->rm', par, mmap)
        # l1 'regions x cbcomb'
        #l1 =  par * M
        #l1 = tf.einsum('rm,cm->rm',par, M)
        #l1 = tf.einsum('rk,mk->rm',self.W, xc)
        #l2 = tf.einsum('rb,mb->rm', self.B, xb)
        #l3 = tf.einsum('r,m->rm', self.R, xr)

        return tf.math.sigmoid(l1 + self.a + self.b)

    def predict(self, adata):
        X, _ = self._prepare_data(adata)
        return self._predict(X)

    def update(self, inputs, y):

        with tf.GradientTape() as tape:
            tot_loss = self._loss(y, self._predict(inputs))

        g = tape.gradient(tot_loss, self.params)
        self.optimizer.apply_gradients(zip(g, self.params))

        return tot_loss

    def _prepare_data(self, adata):

        clusters = adata.obsm[self.cluster]
        nbatches = len(self.batches)
        batches = np.concatenate([adata.obsm[batch] for batch in self.batches], axis=1)
        
        #c x m
        M = np.concatenate([clusters, batches],axis=1)

        def _build_featuremap(i, n):
            x = np.zeros(n)
            x[i]=1
            return x

        cs=[_build_featuremap(c, clusters.shape[1]) for c in  range(clusters.shape[1])]       
        bs=[[_build_featuremap(bi, adata.obsm[batch].shape[1]) for bi in range(adata.obsm[batch].shape[1])] for batch in self.batches]
        
        Mmap = np.asarray([np.concatenate(els) for els in product(cs,*bs)])
            
        # m x m

        # c x m  cell to cluster/batch combination
        M2 = M.dot(Mmap.T) - nbatches
        M2[M2<0] = 0

        # region by cells x cells by M
        y_pos = adata.X.T.dot(M2)
        y_neg = M2.sum(0,keepdims=True) - y_pos

        ## m x b

        X = Mmap
        y = (y_pos,y_neg)
        return X, y

    def fit(self, adata, epochs=10):
        X, y = self._prepare_data(adata)

        for i in range(epochs):
            loss = self.update(X, y)
            #print(f' e {i}: {loss}')


def get_variable_regions(adata, groupby="louvain",
                         batches=['batch'], niter=1000):
    """Extract variable regions relative to clustering."""

    if 'readdepth' not in adata.obs.columns:
        adata.obs['rdepth'] = np.asarray(adata.X.sum(1)).flatten()

    if batches is None:
        batches = ['batch']
        adata.obs.loc[:,'batch'] = pd.Categorical(['dummy']*adata.shape[0])
    
    for batch in batches or []:
        if batch not in adata.obsm:
            cats = adata.obs[batch].cat.categories.tolist()
            onehot = np.zeros((adata.shape[0], len(cats)))
            for ib, cat in enumerate(cats):
                onehot[adata.obs[batch].values==cat, ib]=1
            adata.obsm[batch] = onehot
    if groupby not in adata.obsm:
        cats = adata.obs[groupby].cat.categories.tolist()
        onehot = np.zeros((adata.shape[0], len(cats)))
        for ib, cat in enumerate(cats):
            onehot[adata.obs[groupby].values==cat, ib]=1
        adata.obsm[groupby] = onehot

    mlr = MultiLogisticRegression(adata, groupby, batches)
    mlr.fit(adata, epochs=niter)
    scores = mlr.W.numpy()
    np.testing.assert_equal(scores.shape, (adata.shape[1], adata.obsm[groupby].shape[1]))

    adata.varm['cluster_region'] = mlr.W.numpy()
    adata.varm['batch_region'] = mlr.B.numpy()
    adata.uns['cluster_intercept'] = mlr.b.numpy()
    adata.uns['region_intercept'] = mlr.a.numpy()

    return adata

def get_most_variable_regions(adata, 
                              cluster_regions='cluster_region',
                              groupby='louvain',
                              nregions=100):
    R = adata.varm[cluster_regions]
    
    clusters = adata.obs[groupby].cat.categories
    ids = {}
    for i, cluster in enumerate(clusters):
        rth = np.sort(R[:,i])[::-1][nregions]

        ids[cluster] = adata.var.iloc[list(np.where(R[:,i]>=rth)[0])].index.tolist()
    
    return ids
