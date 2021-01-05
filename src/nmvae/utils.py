import os
import sys
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from scipy.sparse import issparse, coo_matrix
from sklearn.preprocessing import OneHotEncoder
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
    
    #ds = ds.batch(batch_size).map(lambda x: tf.sparse.to_dense(x))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds

class ExpandDims(layers.Layer):
    def __init__(self, axis=1, *args, **kwargs):
        super(ExpandDims, self).__init__(*args, **kwargs)
        self.axis = axis
    def call(self, inputs):
        o = tf.expand_dims(inputs, axis=self.axis)
        return tf.expand_dims(inputs, axis=self.axis)
    def get_config(self):
        config = {'axis':self.axis}
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        


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
        dim = tf.shape(z_mean)[-1]
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

class BatchLoss(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BatchLoss, self).__init__(*args, **kwargs)
        self.catmet = tf.keras.metrics.CategoricalAccuracy(name='bacc')
        self.binmet = tf.keras.metrics.AUC(name='bauc')

    def call(self, inputs):
        if len(inputs) < 2:
            return inputs
        pred_batch, true_batch = inputs
        tf.debugging.assert_non_negative(pred_batch)
        tf.debugging.assert_non_negative(true_batch)
        loss = 0.0
        for tb, pb in zip(true_batch, pred_batch):
            loss += tf.reduce_sum(-tf.math.xlogy(tb, pb+1e-9))
            self.add_metric(self.catmet(tb,pb))
            self.add_metric(self.binmet(tb[:,0,0],pb[:,-1,0]))
        tf.debugging.check_numerics(loss, "targets * log(p)")
        
        self.add_loss(loss)
        return pred_batch

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ScalarBiasLayer(keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(1,),
                                    initializer='ones',
                                    trainable=True)
    def call(self, x):
        return tf.ones((tf.shape(x)[0],) + (1,)*(len(x.shape.as_list())-1))*self.bias


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
                          'ClipLayer': ClipLayer,
                          'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                          'AddBiasLayer': AddBiasLayer,
                          'ScalarBiasLayer':ScalarBiasLayer,
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

    def test_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            data = data[0]
        z = self.encoder(data)
        pred = self.decoder([z, data])

        total_loss = sum(self.encoder.losses) + sum(self.decoder.losses)

        losses['loss'] = total_loss
        return losses

    def predict_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z = self.encoder(data)
        pred = self.decoder([z, data])
        return pred

class BCVAE(keras.Model):
    def __init__(self, encoder, decoder, batcher, **kwargs):
        super(BCVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = keras.Model(self.encoder.inputs,
                                           self.encoder.get_layer('z_mean').output)

        self.batcher = batcher
        self.decoder = decoder
        self.cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    def save(self, filename):
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'
        self.encoder.save(f + '_encoder_' + s)
        self.decoder.save(f + '_decoder_' + s)
        self.batcher.save(f + '_batcher_' + s)

    @classmethod
    def create(cls, params, _create_encoder, _create_decoder, _create_batcher):
         encoder = _create_encoder(params)
         decoder = _create_decoder(params)
         batcher = _create_batcher(params)

         return cls(encoder, decoder, batcher)

    @classmethod
    def load(cls, filename):
        f = filename.split('.h5')[0]
        s='.h5'
        
        custom_objects = {'Sampling': Sampling,
                          'KLlossLayer': KLlossLayer,
                          'ClipLayer': ClipLayer,
                          'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                          'AddBiasLayer': AddBiasLayer,
                          'ScalarBiasLayer':ScalarBiasLayer,
                         }
        encoder = load_model(f + '_encoder_' + s, custom_objects=custom_objects)
        decoder = load_model(f + '_decoder_' + s, custom_objects=custom_objects)
        batcher = load_model(f + '_batcher_' + s, custom_objects=custom_objects)
        return cls(encoder, decoder, batcher)

    def save_weights(self, filename, overwrite=True, save_format=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.save_weights(f + '_encoder_' + s)
        self.decoder.save_weights(f + '_decoder_' + s)
        self.batcher.save_weights(f + '_batcher_' + s)

    def load_weights(self, filename, by_name=False, skip_mismatch=False):
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.load_weights(f + '_encoder_' + s)
        self.decoder.load_weights(f + '_decoder_' + s)
        self.batcher.load_weights(f + '_batcher_' + s)

    def call(self, data):
        if isinstance(data, tuple):
            data, labels = data
        z = self.encoder(data)
        batchpred = self.batcher(z)
        if len(self.decoder.losses) > 0:
            pred = self.decoder([z, data, labels])
        else:
            pred = self.decoder(z)

        return pred, batchpred

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.batcher.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            data, labels = data
        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder(data)
            for i, loss in enumerate(self.encoder.losses):
                losses[f'kl_loss_{i}'] = loss
            pred = self.decoder([z, data, labels])
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            batchpred = self.batcher(z)
            if not isinstance(batchpred, tuple):
                batchpred = (batchpred,)
            batch_loss = []
            for i, (pred, labs) in enumerate(zip(batchpred, labels)):
                batch_loss.append(self.cce(labs, tf.math.reduce_mean(pred, axis=1)))
                losses[f'batch_loss_{i}'] = batch_loss[-1]

            batch_loss = sum(batch_loss)
            total_loss = sum(self.encoder.losses) + sum(self.decoder.losses) - batch_loss

        grads = tape.gradient(total_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))

        grads = tape.gradient(batch_loss, self.batcher.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.batcher.trainable_weights))

        del tape
        losses['loss'] = total_loss
        return losses


    def test_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            data, labels = data
        z = self.encoder(data)
        pred = self.decoder([z, data, labels])
        batchpred = self.batcher(z)

        if not isinstance(batchpred, tuple):
            batchpred = (batchpred,)
        batch_loss = []
        for i, (pred, labs) in enumerate(zip(batchpred, labels)):
            batch_loss.append(self.cce(labs, tf.math.reduce_mean(pred, axis=1)))

        batch_loss = sum(batch_loss)
        total_loss = sum(self.encoder.losses) + sum(self.decoder.losses) - batch_loss

        losses['loss'] = total_loss
        return losses


    def predict_step(self, data):
        data = data[0]
        if isinstance(data, tuple):
            data, labels = data
        z = self.encoder(data)
        pred = self.decoder([z, data, labels])
        return pred


class BCVAE2(keras.Model):
    """ Variational auto-encoder using the batch label as input condition"""
    def __init__(self, encoder, decoder, **kwargs):
        super(BCVAE2, self).__init__(**kwargs)
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
                          'ClipLayer': ClipLayer,
                          'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                          'AddBiasLayer': AddBiasLayer,
                          'ScalarBiasLayer':ScalarBiasLayer,
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
            data, labels = data
        z = self.encoder([data, labels])
        if len(self.decoder.losses) > 0:
            pred = self.decoder([z, data, labels])
        else:
            pred = self.decoder(z)

        return pred

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            data, labels = data
        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder([data, labels])
            for i, loss in enumerate(self.encoder.losses):
                losses[f'kl_loss_{i}'] = loss
            pred = self.decoder([z, data, labels])
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            total_loss = sum(self.encoder.losses) + sum(self.decoder.losses)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        losses['loss'] = total_loss
        return losses


    def test_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            data, labels = data
        z = self.encoder([data, labels])
        pred = self.decoder([z, data, labels])

        total_loss = sum(self.encoder.losses) + sum(self.decoder.losses)

        losses['loss'] = total_loss
        return losses


    def predict_step(self, data):
        data = data[0]
        if isinstance(data, tuple):
            data, labels = data
        z = self.encoder([data, labels])
        pred = self.decoder([z, data, labels])
        return pred

class BCVAE3(keras.Model):
    """ Variational auto-encoder using batch-adversarial training."""
    def __init__(self, encoder, decoder, **kwargs):
        super(BCVAE3, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = keras.Model(self.encoder.get_layer('input_data').input,
                                           self.encoder.get_layer('z_mean').output)

        self.decoder = decoder

        self.encoder_params = keras.Model(self.encoder.get_layer('input_data').input,
                                          self.encoder.get_layer('random_latent').output).trainable_weights
        self.batch_params = [w for w in encoder.trainable_weights if 'batchcorrect' in w.name]

        ba = [l.output for l in self.encoder.layers if 'combine_batches' in l.name]
        test_encoder = keras.Model(self.encoder.get_layer('input_data').input,
                                   [self.encoder.get_layer('random_latent').output, ba], name="encoder")
                                    #self.encoder.get_layer('concatenate').output], name="encoder")
        self.test_encoder = test_encoder

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
                          'ClipLayer': ClipLayer,
                          'BatchLoss': BatchLoss,
                          'BatchKLLoss': BatchKLLoss,
                          'ExpandDims': ExpandDims,
                          'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                          'AddBiasLayer': AddBiasLayer,
                          'ScalarBiasLayer':ScalarBiasLayer,
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
            data, labels = data
        z = self.encoder([data, labels])
        if len(self.decoder.losses) > 0:
            pred = self.decoder([z, data, labels])
        else:
            pred = self.decoder(z)

        return pred

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            profile, labels = data

        with tf.GradientTape(persistent=True) as tape:
            z, b = self.encoder(data)
            kl_loss, batch_loss = self.encoder.losses
            print(self.encoder.losses)
            losses['kl_loss'] = kl_loss
            losses['bloss'] = batch_loss
            #losses['batch_skl'] = bskl
            #losses['batch_kl'] = bkl

            pred = self.decoder([z, profile, labels])
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            recon_loss = sum(self.decoder.losses)
            total_loss = kl_loss + recon_loss - batch_loss

        grads = tape.gradient(total_loss, self.encoder_params + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder_params + self.decoder.trainable_weights))

        grads = tape.gradient(batch_loss, self.batch_params)
        self.optimizer.apply_gradients(zip(grads, self.batch_params))

        del tape
        losses['loss'] = total_loss
        for mn, s in zip(self.encoder.metrics_names, self.encoder.metrics):
            losses[mn] = s.result()
        return losses


    def test_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            profile, labels = data
        #z = self.encoder(data)
        z, b = self.encoder(data)
        #kl_loss, batch_loss = self.encoder.losses
        kl_loss, batch_loss = self.encoder.losses
        pred = self.decoder([z, profile, labels])
        recon_loss = sum(self.decoder.losses)

        total_loss = kl_loss + recon_loss - batch_loss

        losses['loss'] = total_loss
        for mn, s in zip(self.encoder.metrics_names, self.encoder.metrics):
            losses[mn] = s.result()
        return losses


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
