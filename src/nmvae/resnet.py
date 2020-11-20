import os
import shutil
from collections import OrderedDict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger

from scipy.stats import iqr
from sklearn.model_selection import train_test_split
from nmvae.utils import to_dataset, to_sparse, load_data
from nmvae.utils import VAE
from nmvae.utils import ClipLayer
from nmvae.utils import KLlossLayer
from nmvae.utils import Sampling
from nmvae.utils import ScalarBiasLayer
from nmvae.utils import AddBiasLayer
from nmvae.utils import NegativeMultinomialEndpoint
from nmvae.countmatrix import CountMatrix


def load_data(data, regions, cells):
    cmat = CountMatrix.from_mtx(data, regions, cells)
    cmat = cmat.filter(binarize=True)
    cm=cmat.cmat.T.tocsr().astype('float32')

    tokeep = np.asarray(cm.sum(0)).flatten()
    x_data = cm.tocsc()[:,tokeep>0].tocsr()

    rownames = cmat.cannot.barcode
    colnames = cmat.regions.apply(lambda row: f'{row.chrom}_{row.start}_{row.end}',axis=1)

    return x_data, rownames, colnames


def resnet_vae_params(input_dims, args):
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
        ('datadims', input_dims),
        ('latentdims', args.nlatent),
      ]
    )
    return params

def create_repeat_encoder(params):
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs
    x = RepeatVector(params['nrepeat'])(x)

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = RepeatDense1D(nhidden_e)(xinit)
    xinit = layers.Activation(activation='relu')(xinit)
    for _ in range(params['nlayers_e']):
        x = RepeatDense1D(nhidden_e)(xinit)
        x = layers.Activation(activation='relu')(x)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = RepeatDense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    x = xinit
    z_mean = RepeatDense1D(latent_dim, name="z_mean")(x)
    z_log_var = RepeatDense1D(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder


def create_encoder(params):
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs

    xinit = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
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


def create_decoder(params):

    nsamples = params['nsamples']
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')

    x = latent_inputs

    for nhidden in range(params['nlayers_d']):
        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
        x = layers.Dropout(params['hidden_d_dropout'])(x)

    target_inputs = keras.Input(shape=input_shape, name='targets')

    targets = layers.Reshape((1, params['datadims']))(target_inputs)

    # multinomial part
    logits = layers.Dense(params['datadims'],
                          activation='linear', name='logits',
                          use_bias=False)(x)

    logits = AddBiasLayer(name='extra_bias')(logits)

    # dispersion parameter
    r = ScalarBiasLayer()(x)
    r = layers.Activation(activation=tf.math.softplus)(r)
    r = ClipLayer(1e-10, 1e5)(r)

    prob_loss = NegativeMultinomialEndpoint()([logits, r, targets])

    decoder = keras.Model([latent_inputs, target_inputs],
                           prob_loss, name="decoder")

    return decoder

class MetaVAE:
    def __init__(self, params, repeats, output, overwrite):
        self.input_dim = params['datadims']
        self.nlatent = params['latentdims']
        self.repeats = repeats
        self.output = output
        self.models = []
        self.joined_model = None
        self.overwrite = overwrite
        if os.path.exists(output) and overwrite:
            shutil.rmtree(output)
        
        os.makedirs(output, exist_ok=True)
        self.space = params
        #resnet_vae_params(input_dim, nlatent)
        

    def fit(self, x_data,
            shuffle=True, batch_size=64,
            epochs=1, validation_split=.15
           ):

        space = self.space
        x_train, x_test = train_test_split(x_data, test_size=validation_split,
                                           random_state=42)

        tf_X = to_dataset(to_sparse(x_train), shuffle=shuffle, batch_size=batch_size)

        output_bias = np.asarray(x_train.sum(0)).flatten()
        output_bias /= output_bias.sum()
        output_bias = np.log(output_bias + 1e-8)
        
        for r in range(self.repeats):

            subpath = os.path.join(self.output, f'repeat_{r+1}')
            os.makedirs(subpath, exist_ok=True)
            if not os.path.exists(os.path.join(subpath, 'model')):

                print(f'Run repetition {r+1}')
                model = VAE.create(space, create_encoder, create_decoder)

                # initialize the output bias based on the overall read coverage
                # this slightly improves results
                model.decoder.get_layer('extra_bias').set_weights([output_bias])

                model.summary()
                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True)
                             )
                csvcb = CSVLogger(os.path.join(subpath, 'train_summary.csv'))

                model.fit(tf_X, epochs = epochs,
                          validation_data=(to_dataset(to_sparse(x_test), shuffle=False),),
                          callbacks=[csvcb])
                model.save(os.path.join(subpath, 'model', 'vae.h5'))
                self.models.append(model)
            else:
                model = VAE.load(os.path.join(subpath, 'model', 'vae.h5'))
                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True)
                             )
                self.models.append(model)
            model.summary()

    def determine_outliers_threshold(self, data, batch_size=64):
        tf_x_data = to_dataset(to_sparse(data), shuffle=False)
        performance = []
        for _, model in enumerate(models):
            perf = model.evaluate(tf_x_data)
            performance.append(perf)

        performance = np.asarray(performance)
        max_loss = np.quantile(performance, .75) + 1.5* iqr(performance)
        self.max_loss = max_loss

    def get_models(self):
        if len(self.models) < self.repeats:
            for r in range(self.repeats):
                model = VAE.load(os.path.join(self.output, f'repeat_{r+1}', 'model.h5'))
                self.models.append(model)

    def encode(self, data, barcode, batch_size=64):
        tf_x = to_dataset(to_sparse(data), shuffle=False, batch_size=batch_size)

        performance = []
        for model in self.models:
            perf = model.evaluate(tf_x)
            performance.append(perf)

        performance = np.asarray(performance)
        max_loss = np.quantile(performance, .75) + 1.5* iqr(performance)

        dfs = []
        for i, model in enumerate(self.models):
            if performance[i] > max_loss:
                # skip outlier
                continue
            predmodel = keras.Model(model.encoder.inputs, model.encoder.get_layer('z_mean').output)
            out = predmodel.predict(tf_x)
            df = pd.DataFrame(out, index=barcode, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        df.to_csv(os.path.join(self.output, 'latent.csv'))

