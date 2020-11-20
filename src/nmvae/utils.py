import copy
import sys
import traceback
import math
import tensorflow as tf
from umap import UMAP
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pyreadr
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops
import kerastuner as kt
from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner import HyperParameters
from kerastuner.engine.hypermodel import HyperModel
import os
import keras
from scregseg.countmatrix import CountMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy.io import mmread
from scipy.sparse import issparse, coo_matrix, hstack
from scipy.stats import iqr
from collections import OrderedDict

from keras.models import load_model

def to_sparse(x):
    if issparse(x):
        smat = x.tocoo()
    else:
        smat = coo_matrix(x)
    return smat

def to_sparse_tensor(x):
    return tf.SparseTensor(indices=np.mat([x.row, x.col]).transpose(), values=x.data, dense_shape=x.shape)

def to_dataset(x, y=None, batch_size=64, shuffle=True):
    ds_x = tf.data.Dataset.from_tensor_slices(to_sparse_tensor(x))

    if y is not None:
        ds_y = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((ds_x, ds_y))
    else:
        ds = ds_x

    if shuffle:
        ds = ds.shuffle(batch_size*8)
    
    ds = ds.batch(batch_size).map(lambda x: tf.sparse.to_dense(x))
    ds = ds.prefetch(8)
    return ds


def load_data(name, return_all=False):
    if name == 'buenrostro2018':
        name = '/local/wkopp/scATAC-benchmarking/Real_Data/Buenrostro_2018/input/filtered_binarycountmatrix.mtx'
    elif name == 'buenrostro2018bulkpeaks':
        name = '/local/wkopp/scATAC-benchmarking/Real_Data/Buenrostro_2018_bulkpeaks/input/filtered_binarycountmatrix.mtx'
    elif name == '10x':
        name = '/local/wkopp/scATAC-benchmarking/Real_Data/10x_PBMC_5k/input/filtered_binarycountmatrix.mtx'
    elif name == 'cusanovich2018':
        name = '/local/wkopp/scATAC-benchmarking/Real_Data/Cusanovich_2018/input/filtered_binarycountmatrix.mtx'
    elif name == 'cusanovich2018subset':
        name = '/local/wkopp/scATAC-benchmarking/Real_Data/Cusanovich_2018_subset/input/filtered_binarycountmatrix.mtx'
    elif name == 'bonemarrow_clean':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_clean/input/bonemarrow_clean.mtx'
    elif name == 'bonemarrow_cov1000':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov1000/input/bonemarrow_cov1000.mtx'
    elif name == 'bonemarrow_cov250':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov250/input/bonemarrow_cov250.mtx'
    elif name == 'bonemarrow_cov2500':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov2500/input/bonemarrow_cov2500.mtx'
    elif name == 'bonemarrow_cov500':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov500/input/bonemarrow_cov500.mtx'
    elif name == 'bonemarrow_cov5000':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov5000/input/bonemarrow_cov5000.mtx'
    elif name == 'bonemarrow_noisy_p2':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_noisy_p2/input/bonemarrow_noisy_p2.mtx'
    elif name == 'bonemarrow_noisy_p4':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_noisy_p4/input/bonemarrow_noisy_p4.mtx'
    elif name == 'erythropoiesis_clean':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/Erythropoiesis_clean/input/erythropoiesis_clean.mtx'
    elif name == 'erythropoiesis_noisy_p2':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/Erythropoiesis_noisy_p2/input/erythropoiesis_noisy_p2.mtx'
    elif name == 'erythropoiesis_noisy_p4':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/Erythropoiesis_noisy_p4/input/erythropoiesis_noisy_p4.mtx'

    elif name == 'erythropoiesis_clean_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/Erythropoiesis_clean_2kcells/input/erythropoiesis_clean_2kcells.mtx'
    elif name == 'erythropoiesis_noisy_p2_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/Erythropoiesis_noisy_p2_2kcells/input/erythropoiesis_noisy_p2_2kcells.mtx'
    elif name == 'erythropoiesis_noisy_p4_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/Erythropoiesis_noisy_p4_2kcells/input/erythropoiesis_noisy_p4_2kcells.mtx'
    elif name == 'erythropoiesis_noisy_p6_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/Erythropoiesis_noisy_p6_2kcells/input/erythropoiesis_noisy_p6_2kcells.mtx'

    elif name == 'bonemarrow_clean_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_clean_2kcells/input/bonemarrow_clean_2kcells.mtx'
    elif name == 'bonemarrow_cov1000_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov1000_2kcells/input/bonemarrow_cov1000_2kcells.mtx'
    elif name == 'bonemarrow_cov250_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov250_2kcells/input/bonemarrow_cov250_2kcells.mtx'
    elif name == 'bonemarrow_cov2500_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov2500_2kcells/input/bonemarrow_cov2500_2kcells.mtx'
    elif name == 'bonemarrow_cov500_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov500_2kcells/input/bonemarrow_cov500_2kcells.mtx'
    elif name == 'bonemarrow_cov5000_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_cov5000_2kcells/input/bonemarrow_cov5000_2kcells.mtx'
    elif name == 'bonemarrow_noisy_p2_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_noisy_p2_2kcells/input/bonemarrow_noisy_p2_2kcells.mtx'
    elif name == 'bonemarrow_noisy_p4_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_noisy_p4_2kcells/input/bonemarrow_noisy_p4_2kcells.mtx'
    elif name == 'bonemarrow_noisy_p6_2kcells':
        name = '/local/wkopp/scATAC-benchmarking/Synthetic_Data/BoneMarrow_noisy_p6_2kcells/input/bonemarrow_noisy_p6_2kcells.mtx'
    #elif name == 'erythropoiesis_noisy_p4':
    if name == '10x_scrnaseq':
        basedir = '/local/wkopp/scATAC-benchmarking/Real_Data/10x_scRNAseq'
        
        celltypes = ['b_cells',
                 'cd14_monocytes',
                 'cd34_cells',
                 'cd4_helper_tcells',
                 'cd56_nkcells',
                 'cytotoxic_tcells',
                 'memory_tcells',
                 'naive_cytotoxic_tcells',
                 'naive_tcells',
                 'regulatory_tcells',
               ]
        
        mats = []
        labels = []
        for celltype in celltypes:
            print(celltype)
            M = mmread(os.path.join(basedir, 'input', celltype, 'hg19', 'matrix.mtx'))
            mats.append(M)
            labels += [celltype]*M.shape[1]
        
        df = pd.DataFrame({'labelname':labels})
        mapping = {k:i for i,k in enumerate(celltypes)}
        df.loc[:, 'label'] = df.labelname.map(mapping)
        df.loc[:, "barcode"] = np.arange(df.shape[0])
        y_data = df
        
        cnt = hstack(mats).T
        x_data = cnt.tocsr().astype('float32')
        
        if return_all:
            x_train, x_test, y_train, y_test = x_data, x_data, y_data, y_data
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                            test_size=0.15,
                                                            random_state=42)
        x_train=to_sparse(x_train)
        x_test=to_sparse(x_test)
        x_data = to_sparse(x_data)

    if name == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0], -1)).astype("float32")/255
        x_test = x_test.reshape((x_test.shape[0], -1)).astype("float32")/255
        x_train = to_sparse(x_train)
        x_test = to_sparse(x_test)
        x_data = x_train
        y_data = y_train

        y_train = pd.DataFrame({'label':y_train})
        y_train.loc[:,'labelname'] = y_train.label
        y_train.loc[:,'barcode'] = np.arange(y_train.shape[0])

        y_test = pd.DataFrame({'label':y_test})
        y_test.loc[:,'labelname'] = y_test.label
        y_test.loc[:,'barcode'] = np.arange(y_test.shape[0])

    if os.path.exists(name):
        cmat = CountMatrix.from_mtx(name)
        cmat = cmat.filter(binarize=True)
        cm=cmat.cmat.T.tocsr().astype('float32')

        tokeep = np.asarray(cm.sum(0)).flatten()
        x_data = cm.tocsc()[:,tokeep>0].tocsr()

        y_data=cmat.cannot.copy()

        mapping = {k:i for i,k in enumerate(y_data.group.unique())}
        y_data.loc[:, 'label'] = y_data.group.map(mapping)
        y_data.loc[:, 'labelname'] = y_data.group

        if return_all:
            x_train, x_test, y_train, y_test = x_data, x_data, y_data, y_data
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                            test_size=0.15,
                                                            random_state=42)
        x_train=to_sparse(x_train)
        x_test=to_sparse(x_test)
        x_data = to_sparse(x_data)

    print(f'''data: x_train={x_train.shape}
       y_train={y_train.shape}
       x_test={x_test.shape}
       y_test={y_test.shape}
''')
    return (x_train, y_train), (x_test, y_test), (x_data, y_data)


class DummyCallback(Callback):
    pass

class ScoreCollectorCallback(Callback):
    def __init__(self, name, data, labels):
        self.name = name
        self.labels = labels
        self.data = to_dataset(data, shuffle=False)
        self.scores = []

    def on_epoch_end(self, epoch, logs=None):
        z_latent = self.model.encoder_predict.predict(self.data)
        score = silhouette_score(z_latent, self.labels.label)
        self.scores.append(score)

        logs[self.name]=score

class RdsWriterCallback(Callback):
    def __init__(self, data, labels, filename):
        self.filename = filename
        self.labels = labels
        self.data = to_dataset(data, shuffle=False)

    def on_train_end(self, logs=None):
        epoch='last'
        z_latent = self.model.encoder_predict.predict(self.data)
        df = pd.DataFrame(z_latent.T, columns=self.labels.barcode)
        filename = self.filename + f'_data.rds'
        print(f'saved to {filename}')
        pyreadr.write_rds(filename, df)

class UmapCallback(Callback):
    def __init__(self, data, labels, filename):
        self.labels = labels
        self.data = to_dataset(data, shuffle=False)
        self.filename = filename
    def run_callback(self, filename):
        z_latent = self.model.encoder_predict.predict(self.data)

        if z_latent.shape[-1] != 2:
            z_latent = UMAP().fit_transform(z_latent)
        fig, ax = plt.subplots(figsize=(12,10))
        df = pd.DataFrame(dict(x=z_latent[:,0],
                          y=z_latent[:,1], labels=self.labels.labelname))
        sns.scatterplot(x='x', y='y', hue='labels', data=df, ax=ax,
                        palette="Set2")
        print(f'saved to {filename}')
        fig.savefig(filename)

    def on_train_end(self, epoch, logs=None):
        self.run_callback(self.filename)

class ClipLayer(layers.Layer):
    def __init__(self, min_value, max_value, *args, **kwargs):
        super(ClipLayer, self).__init__(*args, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    def call(self, inputs):
        return tf.clip_by_value(inputs, clip_value_min=self.min_value,
                                clip_value_max=self.max_value)
    def get_config(self):
        config = {'min_value':self.min_value,
                  'max_value': self.max_value}
        base_config = super(ClipLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, nsamples=1, *args, **kwargs):
        super(Sampling, self).__init__(*args, **kwargs)
        self.nsamples = nsamples
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_mean = tf.expand_dims(z_mean, axis=1)
        z_log_var = tf.expand_dims(z_log_var, axis=1)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, self.nsamples, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    def get_config(self):
        config = {'nsamples':self.nsamples}
        base_config = super(Sampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class KLlossLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        self.add_loss(kl_loss)
        return z_mean, z_log_var

class MSEEndpoint(keras.layers.Layer):
    def call(self, preds):
        targets, preds = preds

        reconstruction_loss = tf.reduce_mean(
            keras.losses.MSE(targets, preds)
        )
        reconstruction_loss *= 28 * 28
        self.add_loss(reconstruction_loss)
        return preds

class BinaryEndpoint(keras.layers.Layer):
    def call(self, preds):
        targets, preds = preds

        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(targets, tf.math.sigmoid(preds) + 1e-10)
        )

        reconstruction_loss *= 28 * 28

        tf.debugging.check_numerics(reconstruction_loss, "encountered NaNs")

        self.add_loss(reconstruction_loss)
        return preds

class PaddingLayer(keras.layers.Layer):
    def __init__(self, factor, *args, **kwargs):
        super(PaddingLayer, self).__init__(*args, **kwargs)
        self.factor = factor

    def call(self, inputs):
        if self.factor == 1:
            return inputs
        pad = int(np.ceil(inputs.shape[-1]/self.factor) * self.factor - inputs.shape[-1])
        inputs_ = tf.pad(inputs, [[0,0], [0, pad]])
        tf.debugging.assert_near(tf.reduce_sum(inputs), tf.reduce_sum(inputs_))
        return inputs_

    def get_config(self):
        config = {'factor':self.factor}
        base_config = super(PaddingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Dropin(keras.layers.Layer):
    def __init__(self, rate, *args, **kwargs):
        super(Dropin, self).__init__(*args, **kwargs)
        self.rate = rate

    def call(self, inputs, training=None):

        def noised():
            return tf.clip_by_value(inputs + K.random_binomial(
                shape=array_ops.shape(inputs),
                p=self.rate,
                dtype=inputs.dtype),
                clip_value_min=0.0, clip_value_max=1.0)

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'rate':self.rate}
        base_config = super(Dropin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DownsamplingLayer(keras.layers.Layer):
    def call(self, inputs):
        inp = tf.reduce_sum(inputs, axis=-1)
        return inp

class MultinomialEndpoint(keras.layers.Layer):
    def call(self, inputs):
        targets, preds = inputs

        reconstruction_loss = -tf.reduce_mean(
                                multinomial_likelihood(targets,
                                                       tf.math.softmax(preds, axis=-1))
                               )

        tf.debugging.check_numerics(reconstruction_loss, "MultinomialEndpoint NaNs")

        self.add_loss(reconstruction_loss)

        return preds


class ScalarBiasLayer(keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(1,),
                                    initializer='ones',
                                    trainable=True)
    def call(self, x):
        return tf.ones((tf.shape(x)[0],) + (1,)*(len(x.shape.as_list())-1))*self.bias
        #batch = tf.shape(z_mean)[0]
        #return tf.expand_dims(tf.reduce_mean(tf.ones_like(x),
        #                                     axis=-1)*self.bias, -1)

class AddBiasLayer(keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight('extra_bias',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + tf.expand_dims(self.bias, 0)

class NegativeMultinomialEndpoint(keras.layers.Layer):
    def call(self, inputs):
        targets = None
        if len(inputs) == 3:
            logits, r, targets = inputs
        else:
            logits, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           negative_multinomial_likelihood(targets, logits, r)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "NegativeMultinomialEndpoint NaN")

        p = softmax1p(logits)
        p0 = softmax1p0(logits)
        return p * r / (p0 + 1e-10)


@tf.function
def multinomial_likelihood(targets, p):
    return tf.reduce_sum(tf.math.xlogy(targets, p+1e-10), axis=-1)

@tf.function
def negative_multinomial_likelihood(targets, logits, r):
    likeli = tf.reduce_sum(tf.math.xlogy(targets, softmax1p(logits)+1e-10), axis=-1)
    tf.debugging.check_numerics(likeli, "targets * log(p)")
    likeli += tf.reduce_sum(tf.math.xlogy(r, softmax1p0(logits) + 1e-10), axis=-1)
    tf.debugging.check_numerics(likeli, "r * log(1-p)")
    likeli -= tf.reduce_sum(tf.math.lgamma(r), axis=-1)
    tf.debugging.check_numerics(likeli, "lgamma(r)")
    return likeli

@tf.function
def softmax1p(x):
    xmax = tf.reduce_max(x, axis=-1, keepdims=True)
    x = x - xmax
    sp = tf.exp(x) / (tf.exp(-xmax)+ tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True))
    tf.debugging.check_numerics(sp, "softmax1p is NaN")
    return sp

@tf.function
def softmax1p0(x):
    xmax = tf.reduce_max(x, axis=-1)
    x = x - tf.reduce_max(x, axis=-1, keepdims=True)
    sp = tf.exp(-xmax)/ (tf.exp(-xmax)+ tf.reduce_sum(tf.exp(x), axis=-1))
    sp = tf.expand_dims(sp, axis=-1)
    tf.debugging.check_numerics(sp, "softmax1p0 is NaN")
    return sp

class CustomTuner(kt.Tuner):
    def __init__(self, epochs, test_data, all_data, 
                 no_earlystopping, batch_size, name,
                 objective, *args, **kwargs):
        super(CustomTuner, self).__init__(*args, **kwargs)
        self.epochs = epochs
        self.x_test, self.y_test = test_data
        self.x_data, self.y_data = all_data
        self.no_earlystopping = no_earlystopping
        self.batch_size = batch_size
        self.objective = objective
        self.name = name 

    def run_trial(self, trial, X):
        path = os.path.join(self.project_dir, 'trial_' + trial.trial_id)

        model = self.hypermodel.build(trial.hyperparameters)

        if self.objective == 'silhouette':
            scorecb = ScoreCollectorCallback('val_sil', self.x_test, self.y_test)
            scoreallcb = ScoreCollectorCallback('sil', self.x_data, self.y_data)
        else:
            scorecb = scoreallcb = DummyCallback()

        csvcb = CSVLogger(os.path.join(path, 'train_summary.csv'))

        figcb = UmapCallback(self.x_test, self.y_test,
                                  os.path.join(path, '2d_embedding.png'))

        rdscb = RdsWriterCallback(self.x_data, self.y_data,
                                  os.path.join(path, f'FM_{trial.trial_id}_{self.name}_vae'))

        if self.no_earlystopping:
            escb = DummyCallback()
        else:
            escb = EarlyStopping(patience=15, restore_best_weights=True)
        try:
            print(f'proceed with fitting of data={X.shape}')
            tf_X = to_dataset(X, batch_size=self.batch_size, shuffle=True)
            model.fit(tf_X, epochs = self.epochs,
                      validation_data=(to_dataset(self.x_test, shuffle=False),),
                      callbacks=[escb,
                                 scorecb, 
                                 scoreallcb,
                                 csvcb,
                                 warmupcb,
                                 rdscb])
            model.save(os.path.join(path, 'best_model.h5'))
            if self.objective == 'silhouette':
                logs = {'silhouette':scoreallcb.scores[-1]}
            else:
                logs = {'loss': model.evaluate(tf_X)}
            self.on_epoch_end(trial, model, 1, logs=logs)
                                                     #'silhouette_total':scoreallcb.scores[-1]})
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
            print('NaNs encountered in model fitting')
            
            self.on_epoch_end(trial, model, 1, logs={'silhouette':-1.,
                                                     'silhouette_total':-1.})

def create_resnetv1_encoder(params):
    """ resnetv1 is used for a large scale parameter search
    """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu",
                        )(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",
                        )(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,
                        )(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv13_encoder(params):
    """ resnetv13 is used for a large scale parameter search
    """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu",
                        )(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",
                        )(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,
                        )(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    x = xinit
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv14_encoder(params):
    """ resnetv14 is used for a large scale parameter search
    additional by-pass layers
    """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu",
                        )(xinit)
    layers_ = [xinit]
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",
                        )(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,
                        )(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)
        layers_.append(xinit)

    #x = xinit
    #x = layers.Concatenate(
    x = layers.Concatenate()(layers_)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv2_encoder(params):
    """ resnetv2 uses the same model template, but extends the search space """
    return create_resnetv1_encoder(params)

def create_resnetv3_encoder(params):
    """ resnet3 implements layer-wise dropout such that only the identity path is used
        for the forward propagation
    """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu",)(xinit)

    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        #x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.Dropout(params['hidden_e_dropout'], noise_shape=(None, 1))(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv4_encoder(params):
    """ resnet4 uses an additional initial dense layer before the resnet modules """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)
    xinit = layers.Dense(params['nhidden_firstlayer'], activation="relu")(xinit)
    xinit = layers.Dropout(params['inputdropout'])(xinit)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv5_encoder(params):
    """ resnet5 uses an additional initial layer
    it differs from resnet4 because of the smaller parameter space.
    the parameter space is a subspace of resnet4 to speedup the search
    """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)
    xinit = layers.Dense(params['nhidden_firstlayer'], activation="relu")(xinit)
    xinit = layers.Dropout(params['inputdropout'])(xinit)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv6_encoder(params):
    """ resnet6 uses elu instead of relu activation"""
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)
    #xinit = layers.Dense(params['nhidden_firstlayer'], activation="elu")(xinit)
    #xinit = layers.Dropout(params['inputdropout'])(xinit)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="elu")(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e,)(xinit)
        #x = #layers.BatchNormalization()(x)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Activation(activation='elu')(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='elu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv7_encoder(params):
    """ resnet 7 uses dropin noise at the inital layer in addition to the dropout noise"""
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)
    xinit = Dropin(params['dropin'])(xinit)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu",)(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv9_encoder(params):
    """ resnet 9 uses dropin noise at the inital layer in addition to the dropout noise"""
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs

    nhidden_e = params['nhidden_e']

    xinit = layers.Dropout(params['inputdropout'])(x)
    xinit = Dropin(params['dropin'])(xinit)

    xinit = layers.Dense(nhidden_e*3, activation="relu",)(xinit)
    xinit = layers.Dropout(2/3.)(xinit)

    xinit = layers.Dense(nhidden_e, activation="relu",)(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv10_encoder(params):
    """ resnetv10 is used for a large scale parameter search
    similar to v1 but with leakyrelu at the beginning and different
    dropout rates.
    """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e,)(xinit)
    if params['init_activation'] == 'leakyrelu':
        xinit = layers.LeakyReLU()(xinit)
    else:
        xinit = layers.Activation(activation='relu')(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_resnetv11_encoder(params):
    """ resnetv11 is used for a large scale parameter search
    similar to v1 but with leakyrelu at the beginning and different
    dropout rates.
    """
    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation='relu')(xinit)
    #if params['init_activation'] == 'leakyrelu':
    #    xinit = layers.LeakyReLU()(xinit)
    #else:
    #    xinit = layers.Activation(activation='relu')(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        x = layers.GaussianNoise(params['hidden_e_g1'])(x)
        x = layers.Dense(nhidden_e,)(x)
        x = layers.GaussianNoise(params['hidden_e_g2'])(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_densenetv1_encoder(params):

    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1

    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs
    #x = PaddingLayer(params['scalefactor'])(encoder_inputs)
    #x = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])), params['scalefactor']))(x)
    #x = DownsamplingLayer()(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu",)(xinit)
    #xlist = [xinit]
    for _ in range(params['nlayers_e']):
        
        #if len(xlist) > 1:
        #    xconcat = layers.Concatenate()(xlist)
        #else:
        #    xconcat = xlist[0]
        x = layers.Dense(nhidden_e, activation="relu",)(xinit)
        #print('xconcat', xconcat)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e,activation='relu')(x)
        #x = layers.Add()([x, xinit])
        #x = layers.Activation(activation='relu')(x)
        xinit = layers.Concatenate()([x, xinit])

    #x = layers.Concatenate()(xlist)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder

def create_decoder(params):

    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    #params['nsamples'] =
    nsamples = params['nsamples']
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']
    modeltype = params['modeltype']

    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')

    x = latent_inputs

    for nhidden in range(params['nlayers_d']):
        
    #for nhidden in params['nhiddendecoder']:
        x = layers.Dense(params['nhiddendecoder'][nhidden], activation="relu")(x)
        x = layers.Dropout(params['hidden_d_dropout'])(x)

    target_inputs = keras.Input(shape=input_shape, name='targets')

    targets = layers.Reshape((1, params['datadims']))(target_inputs)
    #targets = PaddingLayer(params['scalefactor'])(target_inputs)
    #targets = layers.Reshape((int(np.ceil(params['datadims']/params['scalefactor'])),
    #                         params['scalefactor']))(targets)
    #targets = DownsamplingLayer()(targets)

    logits = layers.Dense(int(np.ceil(params['datadims']/params['scalefactor'])),
                          activation='linear', name='logits',
                          use_bias=False)(x)

    logits = AddBiasLayer(name='extra_bias')(logits)
    #logits = PaddingLayer(params['scalefactor'])(logits)
    #logits = layers.Reshape((-1, params['scalefactor']))(logits)
    #logits = DownsamplingLayer()(logits)

    if modeltype == 'mul':
        logits_loss = MultinomialEndpoint(name='endpoint')([targets, logits])

        decoder = keras.Model([latent_inputs, target_inputs],
                              logits_loss, name="decoder")

    elif modeltype == 'binary':
        logits_loss = BinaryEndpoint()([targets, logits])

        decoder = keras.Model([latent_inputs, target_inputs],
                              logits_loss, name="decoder")

    elif modeltype == 'negmul':

        #targets = keras.Input(shape=input_shape, name='targets')
        #p = layers.Dense(input_shape[0], activation='linear')(x)
        r = layers.Dense(1, activation=tf.math.softplus)(x)
        r = ClipLayer(1e-10, 1e5)(r)

        prob = NegativeMultinomialEndpoint(name='endpoint')([logits, r])
        prob_loss = NegativeMultinomialEndpoint()([logits, r, targets])

        decoder = keras.Model([latent_inputs, target_inputs],
                              prob_loss, name="decoder")

    elif modeltype == 'negmul2':

        #targets = keras.Input(shape=input_shape, name='targets')
        #p = layers.Dense(input_shape[0], activation='linear')(x)
        r = ScalarBiasLayer()(x)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)

        prob_loss = NegativeMultinomialEndpoint()([logits, r, targets])
        prob = NegativeMultinomialEndpoint(name='endpoint')([logits, r])

        decoder = keras.Model([latent_inputs, target_inputs],
                              prob_loss, name="decoder")

    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = keras.Model(self.encoder.inputs,
                                           self.encoder.get_layer('z_mean').output)
        self.decoder = decoder

    def save(self, filename):
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'
        self.encoder.save(f + '_encoder_' + s)
        self.decoder.save(f + '_decoder_' + s)

    @classmethod
    def create(cls, params, _create_encoder, _create_decoder):
         encoder = _create_encoder(params)
         decoder = _create_decoder(params)

         return cls(encoder, decoder)

    @classmethod
    def load(cls, filename):
        f = filename.split('.h5')[0]
        s='.h5'
        
        custom_objects = {'Sampling': Sampling,
                          'KLlossLayer': KLlossLayer,
                          'MSEEndpoint': MSEEndpoint,
                          'ClipLayer': ClipLayer,
                          'BinaryEndpoint': BinaryEndpoint,
                          'MultinomialEndpoint': MultinomialEndpoint,
                          'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                          'AddBiasLayer': AddBiasLayer,
                          'ScalarBiasLayer':ScalarBiasLayer,
                          'DownsamplingLayer': DownsamplingLayer,
                          'PaddingLayer': PaddingLayer,
                         }
        encoder = load_model(f + '_encoder_' + s, custom_objects=custom_objects)
        decoder = load_model(f + '_decoder_' + s, custom_objects=custom_objects)
        return cls(encoder, decoder)

    def save_weights(self, filename, overwrite=True, save_format=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.save_weights(f + '_encoder_' + s)
        self.decoder.save_weights(f + '_decoder_' + s)

    def load_weights(self, filename, by_name=False, skip_mismatch=False):
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.load_weights(f + '_encoder_' + s)
        self.decoder.load_weights(f + '_decoder_' + s)

    def call(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z = self.encoder(data)
        if len(self.decoder.losses) > 0:
            pred = self.decoder([z, data])
        else:
            pred = self.decoder(z)

        return pred

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            for i, loss in enumerate(self.encoder.losses):
                losses[f'kl_loss_{i}'] = loss
            pred = self.decoder([z, data])
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            total_loss = sum(self.encoder.losses) + sum(self.decoder.losses)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        losses['loss'] = total_loss
        return losses

def create_flat_encoder(params):

    if 'scalefactor' not in params or params['modeltype'] == 'binary':
        params['scalefactor'] = 1
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape, name='input_data')

    x = encoder_inputs

    x = layers.Dropout(params['inputdropout'])(x)

    for nhidden in range(params['nlayers_e']):
        
        x = layers.Dense(params['nhiddenencoder'][nhidden], activation="relu",
                        )(x)
        x = layers.Dropout(params['hidden_e_dropout'])(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder


def vae_resnetv1_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [16, 32, 64, 128, 256, 512], default=32)
    nlayers = hp.Int('nlayers_e', 1, 20, default=1)

    max_d_layers = 2

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64, 128], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Int('nlayers_d', 1, max_d_layers, default=1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Int('nsamples', 1, 3)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Fixed('inputdropout', 0.15)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2, 0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2, 0.3])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['mul', 'negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv12_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Fixed(f'nhidden_e', 512)
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 1

    nhiddendecoder = [hp.Fixed(f'nhidden_d_{nlayer}', 64) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Fixed('nsamples', 1)),
        ('nhiddendecoder', nhiddendecoder),
       # ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Fixed('inputdropout', 0.15)),
        ('hidden_e_dropout', hp.Fixed('hidden_e_dropout', 0.3)),
        ('hidden_d_dropout', hp.Fixed('hidden_d_dropout', 0.3)),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Fixed('modeltype', 'negmul2')),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv2_space(hp, input_dims, latent_dims):
    """ original parameters v2
    here we just used the same model template,
    but increased the number of layers and the number of
    nodes per layer
    """
    nhidden = hp.Choice(f'nhidden_e', [512, 600, 700, 800, 900], default=512)
    nlayers = hp.Int('nlayers_e', 10, 40, default=1)

    max_d_layers = 1

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64], default=32) \
               for nlayer in range(max_d_layers)]

    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Int('nsamples', 1, 3)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Float('inputdropout', 0.00, 0.7, default=0.0)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2, 0.3, 0.4, 0.5])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2, 0.3, 0.4, 0.5])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['mul', 'negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv3_space(hp, input_dims, latent_dims):
    """
    this network uses uses layer dropout to see if
    it improves upon normal element-wise dropout
    """
    nhidden = hp.Choice(f'nhidden_e', [128, 256, 512, 600, 700], default=512)
    nlayers = hp.Int('nlayers_e', 1, 40, default=1)

    max_d_layers = 2

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64, 128], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Int('nlayers_d', 1, max_d_layers, default=1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Int('nsamples', 1, 3)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Float('inputdropout', 0.00, 0.3, default=0.0)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2, 0.3, .4, .5, .6])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2, 0.3])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['mul', 'negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv4_space(hp, input_dims, latent_dims):
    """
    similar to the original archtecture,
    but before the resnet modules, a single broad layer is used

    """
    nhidden = hp.Choice(f'nhidden_e', [64, 128, 256, 512, 600, 800], default=512)
    nlayers = hp.Int('nlayers_e', 1, 20, default=1)

    max_d_layers = 2

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64, 128], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Int('nlayers_d', 1, max_d_layers, default=1)),
        ('nhidden_firstlayer', hp.Choice('nhidden_firstlayer', [256, 512, 1024, 2048, 3072])),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Int('nsamples', 1, 3)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Float('inputdropout', 0.00, 0.3, default=0.0)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2, 0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2, 0.3])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['mul', 'negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv5_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [512])
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 2

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Int('nlayers_d', 1, max_d_layers, default=1)),
        ('nhidden_firstlayer', hp.Choice('nhidden_firstlayer', [512, 1024, 2048, 3072])),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Int('nsamples', 1, 3)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Fixed('inputdropout', 0.15)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.3])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['mul', 'negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv6_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [512])
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 1

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Fixed('nsamples', 1)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Float('inputdropout', 0.00, 0.7, default=0.0)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2, 0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2, 0.3])),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype', ['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv7_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [512])
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 1

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16], default=16) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Fixed('nsamples', 1)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Choice('inputdropout', [0.0, .15, .5, .7])),
        ('dropin', hp.Choice('dropin', [0.0, 0.01, 0.03, .1])),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.3])),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype', ['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv8_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [512])
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 1

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16], default=16) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Fixed('nsamples', 1)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Choice('inputdropout', [.15, .3, .35, .4])),
        ('dropin', hp.Choice('dropin', [0.0, .01, .02])),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [.15, 0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.3])),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype', ['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv9_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [512])
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 1

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16], default=16) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Fixed('nsamples', 1)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Choice('inputdropout', [.3])),
        ('dropin', hp.Choice('dropin', [0.0, .02])),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.0, .05, .1, .15, 0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.3])),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype', ['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_densenetv1_space(hp, input_dims, latent_dims):
    nhidden = hp.Choice(f'nhidden_e', [16, 32, 64, 128, 256, 512, 600], default=32)
    nlayers = hp.Int('nlayers_e', 1, 40, default=2)

    max_d_layers = 2

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64, 128], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Int('nlayers_d', 1, max_d_layers, default=1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', hp.Int('nsamples', 1,3, default=1)),
        ('nhiddendecoder', nhiddendecoder),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Float('inputdropout', 0.00, 0.3, default=0.0)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2, 0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2, 0.3])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['mul', 'negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_flat_space(hp, input_dims, latent_dims):
    max_e_layers = 6
    max_d_layers = 3
    nhiddenencoder = [hp.Choice(f'nhidden_e_{nlayer}',
                      [16, 32, 64, 128, 256, 512, 1024], default=32) \
               for nlayer in range(max_e_layers)]

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64, 128], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_e', hp.Int('nlayers_e', 1, max_e_layers, default=1)),
        ('nlayers_d', hp.Int('nlayers_d', 1, max_d_layers, default=1)),
        ('nhiddenencoder', nhiddenencoder),
        ('nhiddendecoder', nhiddendecoder),
        ('nsamples', 2),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Fixed('inputdropout', 0.15)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2])),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['mul', 'negmul2'])),
          #['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor',  hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_flatv2_space(hp, input_dims, latent_dims):
    max_e_layers = 6
    max_d_layers = 3
    nhiddenencoder = [hp.Choice(f'nhidden_e_{nlayer}',
                      [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], default=32) \
               for nlayer in range(max_e_layers)]

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16, 32, 64, 128, 256], default=32) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_e', hp.Int('nlayers_e', 1, max_e_layers, default=1)),
        ('nlayers_d', hp.Int('nlayers_d', 1, max_d_layers, default=1)),
        ('nhiddenencoder', nhiddenencoder),
        ('nhiddendecoder', nhiddendecoder),
        ('nsamples', 2),
        ('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Fixed('inputdropout', 0.15)),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.00, 0.1, 0.2, .3, .5])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.00, 0.1, 0.2, .3, .5])),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype',
          ['negmul2'])),
          #['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor',  hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv10_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [512, 600, 700])
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 1

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16]) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', 1),
        ('nhiddendecoder', nhiddendecoder),
        #('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Choice('inputdropout', [0.15])),
        ('init_activation', hp.Choice('init_activation', ['relu'])),
        ('hidden_e_dropout', hp.Choice('hidden_e_dropout', [0.3])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.3])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype', ['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

def vae_resnetv11_space(hp, input_dims, latent_dims):
    # original parameter space
    nhidden = hp.Choice(f'nhidden_e', [512])
    nlayers = hp.Fixed('nlayers_e', 20)

    max_d_layers = 1

    nhiddendecoder = [hp.Choice(f'nhidden_d_{nlayer}',
                      [16]) \
               for nlayer in range(max_d_layers)]
    params = OrderedDict(
      [
        ('nlayers_d', hp.Fixed('nlayers_d', 1)),
        ('nhidden_e', nhidden),
        ('nlayers_e', nlayers),
        ('nsamples', 1),
        ('nhiddendecoder', nhiddendecoder),
        #('batch_size', hp.Choice('batch_size', [64])),
        ('inputdropout', hp.Choice('inputdropout', [0.15])),
        ('init_activation', hp.Choice('init_activation', ['relu'])),
        ('hidden_e_g1', hp.Choice('hidden_e_g1', [0.0, 0.001, 0.01, 0.1])),
        ('hidden_e_g2', hp.Choice('hidden_e_g2', [0.0, 0.001, 0.01, 0.1])),
        ('hidden_d_dropout', hp.Choice('hidden_d_dropout', [0.3])),
        #('l1', hp.Float('l1', 0.0, 1e-2, default=0.0)),
        #('l2', hp.Float('l2', 0.0, 1e-2, default=0.0)),
        ('datadims', hp.Fixed('datadims', input_dims)),
        ('trainable_extra_bias', True),
        ('latentdims', hp.Fixed('latentdims', latent_dims)),
        ('modeltype', hp.Choice('modeltype', ['negmul2'])),
        ('dummy', hp.Float('dummy', 0, 10.)),
        ('scalefactor', hp.Fixed('scalefactor', 1)),
      ]
    )
    return params

class VAEFactory:
    def __init__(self, name, input_dims, latent_dims):
        self.name = name
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        print(f'using {name}')

    def get_model(self, hp):
        if 'nmvae-flat-v1-tunning' in self.name:
            space = vae_flat_space(hp, self.input_dims, self.latent_dims)
            #model = build_vae_flat(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_flat_encoder, create_decoder)

        #if 'nmvae-flat-v2-tunning' in self.name:
        if self.name in ['nmvae-flat-v2-tunning', 'nmvae-flat-v2-fixed']:
            space = vae_flatv2_space(hp, self.input_dims, self.latent_dims)
            #model = build_vae_flat(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_flat_encoder, create_decoder)

        if self.name in ['nmvae-resnet-v1-tunning', 'nmvae-resnet-v1-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv1_encoder, create_decoder)

        if self.name in ['nmvae-resnet-v2-tunning',  'nmvae-resnet-v2-fixed']:
            space = vae_resnetv2_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv2_encoder, create_decoder)

        if self.name in ['nmvae-resnet-v3-tunning', 'nmvae-resnet-v3-fixed']:
            space = vae_resnetv3_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv3_encoder, create_decoder)

        if self.name in ['nmvae-resnet-v4-tunning', 'nmvae-resnet-v4-fixed']:
            space = vae_resnetv4_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv4_encoder, create_decoder)

        if self.name in ['nmvae-resnet-v5-tunning', 'nmvae-resnet-v5-fixed']:
            space = vae_resnetv5_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv5_encoder, create_decoder)

        if self.name in ['nmvae-resnet-v6-tunning', 'nmvae-resnet-v6-fixed']:
            space = vae_resnetv6_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv6_encoder, create_decoder)

        if self.name in ['nmvae-resnet-v7-tunning', 'nmvae-resnet-v7-fixed']:
            space = vae_resnetv7_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv7_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v8-tunning', 'nmvae-resnet-v8-fixed']:
            space = vae_resnetv8_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv7_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v9-tunning', 'nmvae-resnet-v9-fixed']:
            space = vae_resnetv9_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv9_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v10-tunning', 'nmvae-resnet-v10-fixed']:
            space = vae_resnetv10_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv10_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v11-tunning', 'nmvae-resnet-v11-fixed']:
            space = vae_resnetv11_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv11_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v13-tunning', 'nmvae-resnet-v13-fixed', 'nmvae-resnet-v15-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv13_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v16-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv13_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v15-tunning', 'nmvae-resnet-v15-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv13_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v13b-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            space['modeltype'] = hp.Choice('modeltype', ['binary', 'mul', 'negmul2'])
            model = VAE.create(space, create_resnetv13_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v13m-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv13_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v13s1-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv13_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v14-tunning', 'nmvae-resnet-v14-fixed']:
            space = vae_resnetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv14_encoder, create_decoder)
 
        if self.name in ['nmvae-resnet-v12-tunning', 'nmvae-resnet-v12-fixed']:
            space = vae_resnetv12_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_resnetv1_encoder, create_decoder)

        if self.name in ['nmvae-densenet-v1-tunning', 'nmvae-densenet-v1-fixed']:
            space = vae_densenetv1_space(hp, self.input_dims, self.latent_dims)
            model = VAE.create(space, create_densenetv1_encoder, create_decoder)
        return model

    def get_hyperparameters(self):
        name = self.name
        hp = HyperParameters()
        hp.Fixed('learning_rate', 0.001)
        
        if name == 'nmvae-resnet-v13s1-fixed':
            hp.Fixed('nsamples', 1)
        else:
            hp.Fixed('nsamples', 2)
        if name == 'nmvae-resnet-v1-tunning':
            hp.Fixed('modeltype', 'negmul2')
        if name == 'nmvae-resnet-v3-tunning':
            hp.Fixed('modeltype', 'negmul2')
        if name == 'nmvae-resnet-v2-tunning':
            hp.Fixed('modeltype', 'negmul2')
        if name == 'nmvae-resnet-v4-tunning':
            pass
        if name == 'nmvae-resnet-v5-tunning':
            pass
        if name == 'nmvae-resnet-v1-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed(f'nhidden_d_0', 16)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.3)
            hp.Fixed('modeltype', 'negmul2'),
        if name == 'nmvae-resnet-v13-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed(f'nhidden_d_0', 16)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.3)
            hp.Fixed('modeltype', 'negmul2'),
        if name == 'nmvae-resnet-v15-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed(f'nhidden_d_0', 32)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.1)
            hp.Fixed('modeltype', 'negmul2'),
        if name == 'nmvae-resnet-v16-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 0)
            #hp.Fixed(f'nhidden_d_0', 16)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.3)
            hp.Fixed('modeltype', 'negmul2'),
        if name == 'nmvae-resnet-v13b-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed(f'nhidden_d_0', 16)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.3)
            hp.Fixed('modeltype', 'binary'),
        if name == 'nmvae-resnet-v13m-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed(f'nhidden_d_0', 16)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.3)
            hp.Fixed('modeltype', 'mul'),
        if name == 'nmvae-resnet-v13s1-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed(f'nhidden_d_0', 16)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.3)
            hp.Fixed('modeltype', 'negmul2'),
            hp.Fixed('nsamples', 1)
            #print('nsamples=1')
        if name == 'nmvae-resnet-v14-fixed':
            hp.Fixed('nhidden_e', 512)
            hp.Fixed('nlayers_e', 20)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed(f'nhidden_d_0', 16)
            hp.Fixed('hidden_e_dropout', 0.3)
            hp.Fixed('hidden_d_dropout', 0.3)
            hp.Fixed('modeltype', 'negmul2'),
        if name == 'nmvae-flat-v1-fixed':
            hp.Fixed('nlayers_e', 4)
            hp.Fixed('nhidden_e_0', 4096)
            hp.Fixed('nhidden_e_1', 1024)
            hp.Fixed('nhidden_e_2', 512)
            hp.Fixed('nhidden_e_3', 512)
            hp.Fixed('nlayers_d', 2)
            hp.Fixed('nhidden_d_0', 32)
            hp.Fixed('nhidden_d_1', 128)
            hp.Fixed('modeltype', 'negmul2')
            hp.Fixed('hidden_e_dropout', 0.1)
            hp.Fixed('hidden_d_dropout', 0.1)
        if name == 'nmvae-resnet-v11-fixed':
            hp.Fixed('hidden_e_g1', 0.0)
            hp.Fixed('hidden_e_g2', 0.001)

        if name == 'nmvae-flat-v1-tunning':
            hp.Fixed('inputdropout', 0.15)
            hp.Fixed('modeltype', 'negmul2')
        if name == 'nmvae-flat-v2-fixed':
            hp.Fixed('hidden_d_dropout', 0.1)
            hp.Fixed('hidden_e_dropout', 0.2)
            hp.Fixed('inputdropout', 0.15)
            hp.Fixed('nlayers_d', 1)
            hp.Fixed('nlayers_e', 5)
            hp.Fixed('nhidden_d_0', 16)
            hp.Fixed('nhidden_e_0', 256)
            hp.Fixed('nhidden_e_1', 16)
            hp.Fixed('nhidden_e_2', 4096)
            hp.Fixed('nhidden_e_3', 1024)
            hp.Fixed('nhidden_e_4', 1024)
            hp.Fixed('nhidden_e_5', 512)
        if name == 'nmvae-densenet-v1-tunning':
            pass
        if name == 'nmvae-resnet-v8-fixed':
            hp.Fixed('inputdropout', 0.15)
            hp.Fixed('dropin', 0.02)
            hp.Fixed('hidden_e_dropout', 0.15)
        return hp


class HyperVAE(HyperModel):
    def __init__(self, name, input_dims, latent_dims, biases=None):
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        if biases is not None:
            self.output_biases = biases
        else:
            self.output_biases = np.ones(input_dims)/input_dims
        self.factory = VAEFactory(name, input_dims, latent_dims)

    def build(self, hp):

        model = self.factory.get_model(hp)

        r = np.log(self.output_biases + 1e-8)
        model.decoder.get_layer('extra_bias').set_weights([r])

        model.compile(optimizer=keras.optimizers.Adam(
                      learning_rate=hp.Float('learning_rate',
                                            sampling='LOG',
                                            min_value=1e-6,
                                            max_value=0.008,
                                            default=0.001),
                      amsgrad=True),
                      run_eagerly=False)
        model.summary()
        return model

def run_keras_tuner(args):

    (x_train, y_train), (x_test, y_test), (x_data, y_data) = load_data(args.data)

    biases = np.asarray(x_train.sum(0)).flatten()
    biases /= biases.sum()

    build_model = HyperVAE(args.model, x_train.shape[-1], args.nlatent, biases)

    hp = build_model.factory.get_hyperparameters()
    if args.objective == 'silhouette':
        objective = kt.Objective('silhouette', 'max')
    else:
        objective = kt.Objective('loss', 'min')

    tuner = CustomTuner(
       epochs=args.epochs,
       test_data=(x_test,y_test),
       all_data=(x_data, y_data),
       no_earlystopping=args.no_earlystopping,
       batch_size=args.batch_size,
       name=args.name,
       objective=args.objective,
       oracle=kt.oracles.BayesianOptimization(
          objective=objective,
          hyperparameters=hp,
          max_trials=args.ntrials),
       hypermodel=build_model,
       overwrite=args.overwrite,
       directory=f'kt_vae_{args.data}_{args.model}_{args.name}',
       project_name=f'vae_{args.name}')

    tuner.search_space_summary()
    if args.no_val:
        tuner.search(x_data)
    else:
        tuner.search(x_train)

    models = tuner.get_best_models(num_models=args.ntrials)

    tuner.results_summary()

    #print(performance)
    # perform model evaluation and detect outliers
    print(x_data.shape)
    tf_x_data = to_dataset(x_data, shuffle=False)
    performance = []
    for _, model in enumerate(models):
        perf = model.evaluate(tf_x_data)
        performance.append(perf)

    performance = np.asarray(performance)
    max_loss = np.quantile(performance, .75) + 1.5* iqr(performance)

    features = []
    new_perf = []
    #reconstruction = np.zeros(x_data.shape)
    n=0
    for i, model in enumerate(models):
        if performance[i] <= max_loss:
            new_perf.append(performance[i])
            feat = model.encoder_predict.predict(tf_x_data)
            features.append(feat)
            filename = os.path.join(f'kt_vae_{args.data}_{args.model}_{args.name}',
                            f'FM_singlerun{i+1}_{args.data}_vae_{args.model}_{args.name}__data.rds')
            pd.DataFrame(feat, index=y_data.barcode).to_csv(filename + '.csv')
       #     recon = model(x_data.toarray())
       #     print(recon.shape)
       #     reconstruction += recon.numpy().mean(1)
       #     n+=1

       # reconstruction /= n
       # df = pd.DataFrame(reconstruction, index=y_data.barcode)
       # filename = os.path.join(f'kt_vae_{args.data}_{args.model}_{args.name}',
       #                         f'recon_merged_{args.data}_vae_{args.model}_{args.name}__data.csv')
       # df.to_csv(filename + '.csv')

#    for i, model in enumerate(models):
#        if performance[i] <= max_loss:
#            new_perf.append(performance[i])
#            feat = model.encoder_predict.predict(tf_x_data)
#            features.append(feat)
#            filename = os.path.join(f'kt_vae_{args.data}_{args.model}_{args.name}',
#                            f'FM_singlerun{i+1}_{args.data}_vae_{args.model}_{args.name}__data.rds')
#            pd.DataFrame(feat, index=y_data.barcode).to_csv(filename + '.csv')

    features_all = np.concatenate(features, axis=1)

    df = pd.DataFrame(features_all, index=y_data.barcode)

    filename = os.path.join(f'kt_vae_{args.data}_{args.model}_{args.name}',
                            f'FM_merged_{args.data}_vae_{args.model}_{args.name}__data.rds')
    print(f'saved {df.shape} features to {filename}')
    df.to_csv(filename + '.csv')
    pyreadr.write_rds(filename, df.T)

    features_best3 = np.concatenate([features[i] for i in np.argsort(new_perf)[:3]], axis=1)

    df = pd.DataFrame(features_best3, index=y_data.barcode)

    filename = os.path.join(f'kt_vae_{args.data}_{args.model}_{args.name}',
                            f'FM_best3_{args.data}_vae_{args.model}_{args.name}__data.rds')
    print(f'saved {df.shape} features to {filename}')
    df.to_csv(filename + '.csv')
    pyreadr.write_rds(filename, df.T)


    features_best5 = np.concatenate([features[i] for i in np.argsort(new_perf)[:5]], axis=1)

    df = pd.DataFrame(features_best5, index=y_data.barcode)

    filename = os.path.join(f'kt_vae_{args.data}_{args.model}_{args.name}',
                            f'FM_best5_{args.data}_vae_{args.model}_{args.name}__data.rds')
    print(f'saved {df.shape} features to {filename}')
    df.to_csv(filename + '.csv')
    pyreadr.write_rds(filename, df.T)


