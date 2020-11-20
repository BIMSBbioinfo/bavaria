
import os
import numpy as np
from scipy.sparse import csr_matrix

from nmvae.cli import main
from nmvae.utils import softmax1p, softmax1p0
from nmvae.utils import SparseSequence
from nmvae.utils import ScalarBiasLayer
from nmvae.utils import PaddingLayer
from nmvae.utils import AddBiasLayer
from nmvae.utils import load_data
from nmvae.utils import VAE, HyperVAE, CustomTuner
from nmvae.utils import create_flat_encoder, create_decoder

import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import kerastuner as kt


def test_softmax1p():
    x=np.random.rand(3,2)

    p = np.exp(x)/(1. + np.exp(x).sum(-1, keepdims=True))
    p0 = 1/(1. + np.exp(x).sum(-1)).reshape(-1,1)

    np.testing.assert_allclose(softmax1p(x).numpy(), p, atol=1e-7)
    np.testing.assert_allclose(softmax1p0(x).numpy(), p0, atol=1e-7)

    np.testing.assert_allclose(softmax1p(x).numpy().sum() + \
                               softmax1p0(x).numpy().sum(), 3.0,
                               atol=1e-7)

def test_sparsesequence_dense():

    X = np.random.rand(3,2)
    seq = SparseSequence(X, shuffle=False)

    np.testing.assert_equal(seq[0], X)

def test_sparsesequence_sparse():

    X = np.random.rand(3,2)
    X = csr_matrix(X)
    seq = SparseSequence(X, shuffle=False)

    np.testing.assert_equal(seq[0], X.todense())

def test_biaslayer():
    xin = keras.Input((2,))
    X = np.random.rand(3,2)
    o = ScalarBiasLayer()(xin)
    m = keras.Model(xin, o)
    pred = m.predict(X)

    np.testing.assert_equal(pred, np.ones_like(pred))

def test_paddinglayer():
    xin = keras.Input((3,))
    X = np.random.rand(3,3)
    Xout = np.zeros((3,4))
    Xout[:,:3]=X
    o = PaddingLayer(2)(xin)
    m = keras.Model(xin, o)
    pred = m.predict(X)

    np.testing.assert_allclose(pred, Xout, atol=1e-7)

def test_addextrabias():

    xin = keras.Input((3,))
    X = np.random.rand(3,3)
    Xout = np.zeros((3,4))
    Xout[:,:3]=X

    o = keras.layers.Dense(2, use_bias=False, name='dense')(xin)
    o = AddBiasLayer(name='extra_bias', trainable=False)(o)
    m = keras.Model(xin, o)

    pred = m.predict(X)

    np.testing.assert_allclose(pred.shape, (3,2), atol=1e-7)

    assert len(m.trainable_weights) == 1
    assert len(m.layers[1].weights) == 1
    assert len(m.layers[2].weights) == 1
    assert m.layers[1].trainable

    #print(m.layers[-1].get_weights()[0])
    #print(np.ones(2))
    m.layers[-1].set_weights([np.ones(2)])
    pred1 = m.predict(X)

    np.testing.assert_allclose(pred.shape, (3,2), atol=1e-7)

    assert len(m.trainable_weights) == 1
    assert len(m.layers[1].weights) == 1
    assert len(m.layers[2].weights) == 1
    assert m.layers[1].trainable

    np.testing.assert_allclose(pred + 1., pred1)

def test_nmvae_flat(tmpdir):
    (x_train, y_train), (x_test, y_test) = load_data('mnist')
    params = {
      'nhidden_e_0': 32,
      'nhidden_e_1': 32,
      'nhidden_d_0': 32,
      'nhidden_d_1': 32,
      'nlayers_e':2,
      'nlayers_d':2,
      'nsamples':1,
      'inputdropout':0.0,
      'nhiddendecoder':[32,32],
      'nhiddenencoder':[32, 32],
      'batch_size': 32,
      'hidden_e_dropout': 0.0,
      'hidden_d_dropout': 0.0,
      'datadims': x_train.shape[-1],
      'trainable_extra_bias': True,
      'latentdims': 3,
      'modeltype': 'negmul2',
      'scalefactor':1
    }
      
    model = VAE.create(params, create_flat_encoder, create_decoder)
    model.compile(optimizer='adam')
    model.fit(x_test[:10])
    model.save('model.h5')
    VAE.load('model.h5')
    model.save(os.path.join(tmpdir.strpath, 'model.h5'))

    model.save_weights(os.path.join(tmpdir.strpath, 'model.h5'))
    model.load_weights(os.path.join(tmpdir.strpath, 'model.h5'))

    model2 = VAE.create(params, create_flat_encoder, create_decoder)
    model2.load_weights(os.path.join(tmpdir.strpath, 'model.h5'))

    np.testing.assert_allclose(model.encoder_predict(x_train[:2]),
                               model2.encoder_predict(x_train[:2]))

def test_nmvae_flat_saveload(tmpdir):
    (x_train, y_train), (x_test, y_test) = load_data('mnist')
    params = {
      'nhidden_e_0': 32,
      'nhidden_e_1': 32,
      'nhidden_d_0': 32,
      'nhidden_d_1': 32,
      'nlayers_e':2,
      'nlayers_d':2,
      'nsamples':1,
      'inputdropout':0.0,
      'nhiddendecoder':[32,32],
      'nhiddenencoder':[32, 32],
      'batch_size': 32,
      'hidden_e_dropout': 0.0,
      'hidden_d_dropout': 0.0,
      'datadims': x_train.shape[-1],
      'trainable_extra_bias': True,
      'latentdims': 3,
      'modeltype': 'negmul2',
      'scalefactor':1
    }
      
    model = VAE.create(params, create_flat_encoder, create_decoder)
    model.compile(optimizer='adam')
    model.fit(x_test[:10])
    model.save(os.path.join(tmpdir.strpath, 'model.h5'))
    model2 = VAE.load(os.path.join(tmpdir.strpath, 'model.h5'))

    np.testing.assert_allclose(model.encoder_predict(x_train[:2]),
                               model2.encoder_predict(x_train[:2]))


def test_hypernmvae_flat(tmpdir):
    (x_train, y_train), (x_test, y_test) = load_data('mnist')
    biases = x_train.sum(0)
    biases /= biases.sum()

    build_model = HyperVAE('nmvae-flat', x_train.shape[-1], 3, biases)
    hp = build_model.factory.get_hyperparameters()

    path = tmpdir.strpath
    tuner = CustomTuner(
        epochs=1,
        test_data=(x_test[:10],y_test[:10]),
        all_data=(x_test[:10], y_test[:10]),
        #oracle=kt.oracles.RandomSearch(
        #   objective=kt.Objective('silhouette', 'max'),
        #   max_trials=200),
        oracle=kt.oracles.BayesianOptimization(
           objective=kt.Objective('silhouette', 'max'),
           hyperparameters=hp,
           max_trials=1),
        #oracle=kt.oracles.Hyperband(
        #   objective=kt.Objective('silhouette', 'max'),
        #   max_trials=200,
        #   max_epochs=200),
        #distribution_strategy=tf.distribute.MirroredStrategy(),
        hypermodel=build_model,
        overwrite=True,
        directory=f'{path}/kt_vae',
        project_name=f'vae')

    tuner.search_space_summary()
    tuner.search(x_train[:10])

    models = tuner.get_best_models(num_models=10)

    tuner.results_summary()


def test_nmvae_flat_saveload_callback(tmpdir):
    (x_train, y_train), (x_test, y_test) = load_data('mnist')
    params = {
      'nhidden_e_0': 32,
      'nhidden_e_1': 32,
      'nhidden_d_0': 32,
      'nhidden_d_1': 32,
      'nlayers_e':2,
      'nlayers_d':2,
      'nsamples':1,
      'inputdropout':0.0,
      'nhiddendecoder':[32,32],
      'nhiddenencoder':[32, 32],
      'batch_size': 32,
      'hidden_e_dropout': 0.0,
      'hidden_d_dropout': 0.0,
      'datadims': x_train.shape[-1],
      'trainable_extra_bias': True,
      'latentdims': 3,
      'modeltype': 'negmul2',
      'scalefactor':1
    }
      
    model = VAE.create(params, create_flat_encoder, create_decoder)
    model.compile(optimizer='adam')
    mcb = ModelCheckpoint(os.path.join(tmpdir.strpath, 'model.h5'), save_weights_only=False)
    model.fit(x_test[:10], callbacks = [mcb])
    #model.save(os.path.join(tmpdir.strpath, 'model.h5'))
    model2 = VAE.load(os.path.join(tmpdir.strpath, 'model.h5'))

    np.testing.assert_allclose(model.encoder_predict(x_train[:2]),
                               model2.encoder_predict(x_train[:2]))


