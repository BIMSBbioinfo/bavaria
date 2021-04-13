import os
from anndata import AnnData
from anndata import read_h5ad
from bavaria.countmatrix import CountMatrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import issparse, coo_matrix
import tensorflow as tf

def load_batch_labels(adata, batches):
    if batches is None:
        df = pd.DataFrame({'dummybatch':['dummy']*len(barcodes)},
                           index=adata.obs.index)
    elif isinstance(batches, str) and os.path.exists(batches):
        df = pd.read_csv(batches, sep='\t', index_col=0)
    return df
    
def one_hot_encode_batches(adata, batchnames):
    for label in batchnames:
        if label not in adata.obsm:
            oh= OneHotEncoder(sparse=False).fit_transform(adata.obs[label].values.astype(str).reshape(-1,1).tolist())
            adata.obsm[label] = oh
    return adata

def load_data(data, regions, cells):
    if data.endswith('.h5ad'):
        return read_h5ad(data)
    cmat = CountMatrix.from_mtx(data, regions, cells)
    cmat = cmat.filter(binarize=True)
    cm=cmat.cmat.T.tocsr().astype('float32')

    tokeep = np.asarray(cm.sum(0)).flatten()
    x_data = cm.tocsc()[:,tokeep>0].tocsr()

    rownames = cmat.cannot.barcode
    
    colnames = cmat.regions.apply(lambda row: f'{row.chrom}_{row.start}_{row.end}',axis=1)

    cmat.regions.loc[:,'name'] = colnames
    cmat.regions.set_index('name', inplace=True)
    obs = cmat.cannot.copy()
    obs.set_index('barcode', inplace=True)
    if 'cell' in obs.columns:
        obs.drop('cell', axis=1, inplace=True)
    adata = AnnData(x_data, 
                    obs=obs,
                    var=cmat.regions)
    adata.obs_names_make_unique()
    return adata


def to_sparse(x):
    if issparse(x):
        smat = x.tocoo()
    else:
        smat = coo_matrix(x)
    return smat


def to_sparse_tensor(x):
    return tf.SparseTensor(indices=np.mat([x.row, x.col]).transpose(), values=x.data, dense_shape=x.shape)


def to_dataset(x, y=None, batch_size=64, shuffle=True):
    ds_x = tf.data.Dataset.from_tensor_slices(to_sparse_tensor(x)).map(lambda x: tf.sparse.to_dense(x))

    if y is not None:
        if isinstance(y, list):
            ds_y = tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensor_slices(d) for d in y]))
        else:
            ds_y = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((ds_x,ds_y))
    else:
        ds = ds_x

    if shuffle:
        ds = ds.shuffle(batch_size*8)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds

