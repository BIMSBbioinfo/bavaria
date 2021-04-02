import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from nmvae.layers import ClipLayer
from nmvae.layers import KLlossLayer
from nmvae.layers import Sampling
from nmvae.layers import ExpandDims
from nmvae.layers import BatchLoss
from nmvae.layers import ScalarBiasLayer
from nmvae.layers import AddBiasLayer
from nmvae.layers import AverageChannel
from nmvae.layers import MutInfoLayer
from nmvae.layers import NegativeMultinomialEndpoint
from nmvae.layers import NegativeMultinomialEndpointV2

def create_encoder(params):
    """ Encoder without batch correction."""
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


def create_encoder_mutinfo(params):
    """ Encoder without batch correction."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs

    #xinit = layers.Dropout(params['inputdropout'])(x)
    xinit = x

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        #x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    x = xinit
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean = MutInfoLayer()(z_mean)
    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder


def create_batch_encoder(params):
    """ Condition on batches at first layer."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')
 
    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    batch_layer = batch_inputs

    if len(batch_layer)>1:
        batch_layer = layers.Concatenate()(batch_layer)
    else:
        batch_layer = batch_layer[0]

    xinit = layers.Concatenate()([xinit, batch_layer])
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

    encoder = keras.Model([encoder_inputs, batch_inputs], z, name="encoder")

    return encoder


def create_batch_encoder_alllayers(params):
    """ Condition on batches in all hidden layers."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')
 
    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    batch_layer = batch_inputs

    if len(batch_layer)>1:
        batch_layer = layers.Concatenate()(batch_layer)
    else:
        batch_layer = batch_layer[0]

    xinit = layers.Concatenate()([xinit, batch_layer])
    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)

    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)

        xbatch = layers.Dense(nhidden_e)(batch_layer)
        x = layers.Add()([x, xinit, xbatch])
        xinit = layers.Activation(activation='relu')(x)

    x = xinit

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model([encoder_inputs, batch_inputs], z, name="encoder")

    return encoder


def create_batch_encoder_gan(params):
    """ With batch-adversarial learning on all hidden layers."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)

    batches = []
    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e)(xinit)
    batches.append(create_batch_net(xinit, params, '00'))
    xinit = layers.Activation(activation='relu')(xinit)
    

    for i in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        batches.append(create_batch_net(x, params, f'1{i}'))
        xinit = layers.Activation(activation='relu')(x)

    x = xinit

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    batches.append(create_batch_net(z, params, f'20'))

    pred_batches = combine_batch_net(batches)

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    true_batch_layer = [ExpandDims()(l) for l in batch_inputs]

    batch_loss = BatchLoss(name='batch_loss')([pred_batches, true_batch_layer])

    encoder = keras.Model([encoder_inputs, batch_inputs], [z, batch_loss], name="encoder")

    return encoder


def create_batch_encoder_gan_lastlayer(params):
    """ With batch-adversarial learning on last latent dims."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)

    batches = []
    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e)(xinit)
    #batches.append(create_batch_net(xinit, params, '00'))
    xinit = layers.Activation(activation='relu')(xinit)
    

    for i in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        #batches.append(create_batch_net(x, params, f'1{i}'))
        xinit = layers.Activation(activation='relu')(x)

    x = xinit

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    batches.append(create_batch_net(z, params, f'20'))

    pred_batches = combine_batch_net(batches)

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    true_batch_layer = [ExpandDims()(l) for l in batch_inputs]

    batch_loss = BatchLoss(name='batch_loss')([pred_batches, true_batch_layer])

    encoder = keras.Model([encoder_inputs, batch_inputs], [z, batch_loss], name="encoder")

    return encoder


def create_batcher(params):
    warnings.warn("create_batcher is experimental and may be removed.",
                  type=DeprecationWarning)

    nsamples = params['nsamples']
    latent_dim = params['latentdims']

    latent_input = keras.Input(shape=(nsamples, latent_dim,), name='input_batcher')

    targets = create_batch_net(latent_input, params, '')
    model = keras.Model(latent_input, targets, name='batcher')

    return model

def create_batchlatent_predictor(params):
    warnings.warn("create_batchlatent_predictor is experimental and may be removed.",
                  type=DeprecationWarning)
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data_2')

    x = encoder_inputs
    for i in range(params['nlayersbatcher']):
       x = layers.Dense(params['nhiddenbatcher'], activation='relu',
                        name=f'batchlatent_hidden_{i}')(x)
       x = layers.BatchNormalization(name=f'batchlatent_batch_norm_{i}')(x)
    x = layers.Dense(params['nlasthiddenbatcher'], activation='relu',
                     name=f'batchlatent_lasthidden_{i}')(x)
    hidden = x
    if len(x.shape.as_list()) <= 2:
        x = ExpandDims()(x)
    pred_batches = [layers.Dense(nl, activation='softmax',
                                 name='batchlatent_out_' + bname)(x) \
                                 for nl,bname in \
                                 zip(params['nbatchcats'], params['batchnames'])]

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    true_batch_layer = [ExpandDims()(l) for l in batch_inputs]

    batch_loss_pred = BatchLoss(name='batch_loss_pred')([pred_batches, true_batch_layer])
    
    predictor = keras.Model([encoder_inputs, batch_inputs], [hidden, batch_loss_pred], name="batch_predictor")
    predictor.summary()

    return predictor

def create_batch_net(inlayer, params, name):
    x = layers.BatchNormalization(name='batchcorrect_batch_norm_1_'+name)(inlayer)
    for i in range(params['nlayersbatcher']):
       x = layers.Dense(params['nhiddenbatcher'], activation='relu', name=f'batchcorrect_{name}_hidden_{i}')(x)
       x = layers.BatchNormalization(name=f'batchcorrect_batch_norm_2_{name}_{i}')(x)
    if len(x.shape.as_list()) <= 2:
        x = ExpandDims()(x)
    targets = [layers.Dense(nl, activation='softmax', name='batchcorrect_'+name + '_out_' + bname)(x) \
               for nl,bname in zip(params['nbatchcats'], params['batchnames'])]
    return targets

def combine_batch_net(batches):
    if len(batches)<=1:
        return batches[0]
    new_output = []
    for bo,_ in enumerate(batches[0]):
        new_output.append(layers.Concatenate(axis=1, name=f'combine_batches_{bo}')([batch[bo] for batch in batches]))
    return new_output

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

#def create_decoder_v20(params):
#    warnings.warn("create_decoder_v20 is experimental and may be removed.",
#                  type=DeprecationWarning)
#
#    nsamples = params['nsamples']
#    input_shape = (params['datadims'],)
#    latent_dim = params['latentdims']
#
#    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
#
#    x = latent_inputs
#
#    for nhidden in range(params['nlayers_d']):
#        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
#        x = layers.Dropout(params['hidden_d_dropout'])(x)
#
#    target_inputs = keras.Input(shape=input_shape, name='targets')
#
#    targets = layers.Reshape((1, params['datadims']))(target_inputs)
#
#    # multinomial part
#    mullogits = layers.Dense(params['datadims'],
#                          activation='linear', name='logits',
#                          use_bias=False)(x)
#
#    mullogits = AddBiasLayer(name='extra_bias')(mullogits)
#
#    # dispersion parameter
#    
#    p0logit = layers.Dense(1)(latent_inputs)
#    r = layers.Dense(1)(latent_inputs)
#    r = layers.Activation(activation=tf.math.softplus)(r)
#    r = ClipLayer(1e-10, 1e5)(r)
#
#    prob_loss = NegativeMultinomialEndpointV2()([mullogits, p0logit, r, targets])
#
#    decoder = keras.Model([latent_inputs, target_inputs],
#                           prob_loss, name="decoder")
#
#    return decoder
#
#def create_decoder_v21(params):
#    warnings.warn("create_decoder_v21 is experimental and may be removed.",
#                  type=DeprecationWarning)
#
#    nsamples = params['nsamples']
#    input_shape = (params['datadims'],)
#    latent_dim = params['latentdims']
#
#    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
#
#    x = latent_inputs
#
#    for nhidden in range(params['nlayers_d']):
#        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
#        x = layers.Dropout(params['hidden_d_dropout'])(x)
#
#    target_inputs = keras.Input(shape=input_shape, name='targets')
#
#    targets = layers.Reshape((1, params['datadims']))(target_inputs)
#    #targets = layers.RepeatVector(nsamples)(targets)
#
#    # multinomial part
#    mullogits = layers.Dense(params['datadims'],
#                          activation='linear', name='logits',
#                          use_bias=False)(x)
#
#    mullogits = AddBiasLayer(name='extra_bias')(mullogits)
#
#    # dispersion parameter
#    x = AverageChannel()(targets)
#    #x = layers.Concatenate(axis=-1)([latent_inputs, targets])
#
#    p0logit = layers.Dense(1)(x)
#    r = layers.Dense(1)(x)
#    r = layers.Activation(activation=tf.math.softplus)(r)
#    r = ClipLayer(1e-10, 1e5)(r)
#
#    prob_loss = NegativeMultinomialEndpointV2()([mullogits, p0logit, r, targets])
#
#    decoder = keras.Model([latent_inputs, target_inputs],
#                           prob_loss, name="decoder")
#
#    return decoder
#
#def create_decoder_v22(params):
#    warnings.warn("create_decoder_v22 is experimental and may be removed.",
#                  type=DeprecationWarning)
#
#    nsamples = params['nsamples']
#    input_shape = (params['datadims'],)
#    latent_dim = params['latentdims']
#
#    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
#
#    x = latent_inputs
#
#    for nhidden in range(params['nlayers_d']):
#        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
#        x = layers.Dropout(params['hidden_d_dropout'])(x)
#
#    target_inputs = keras.Input(shape=input_shape, name='targets')
#
#    targets = layers.Reshape((1, params['datadims']))(target_inputs)
#
#    # multinomial part
#    mullogits = layers.Dense(params['datadims'],
#                          activation='linear', name='logits',
#                          use_bias=False)(x)
#
#    mullogits = AddBiasLayer(name='extra_bias')(mullogits)
#
#    # dispersion parameter
#    #x = AverageChannel()(targets)
#    #x = layers.Concatenate(axis=-1)([latent_inputs, targets])
#    x = layers.Dense(params['nhiddendecoder'], activation='relu')(targets)
#
#    p0logit = layers.Dense(1)(x)
#    r = layers.Dense(1)(x)
#    r = layers.Activation(activation=tf.math.softplus)(r)
#    r = ClipLayer(1e-10, 1e5)(r)
#
#    prob_loss = NegativeMultinomialEndpointV2()([mullogits, p0logit, r, targets])
#
#    decoder = keras.Model([latent_inputs, target_inputs],
#                           prob_loss, name="decoder")
#
#    return decoder
#
#def create_decoder_v23(params):
#    warnings.warn("create_decoder_v23 is experimental and may be removed.",
#                  type=DeprecationWarning)
#
#    nsamples = params['nsamples']
#    input_shape = (params['datadims'],)
#    latent_dim = params['latentdims']
#
#    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
#
#    x = latent_inputs
#
#    for nhidden in range(params['nlayers_d']):
#        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
#        x = layers.Dropout(params['hidden_d_dropout'])(x)
#
#    target_inputs = keras.Input(shape=input_shape, name='targets')
#
#    targets = layers.Reshape((1, params['datadims']))(target_inputs)
#
#    # multinomial part
#    mullogits = layers.Dense(params['datadims'],
#                          activation='linear', name='logits',
#                          use_bias=False)(x)
#
#    mullogits = AddBiasLayer(name='extra_bias')(mullogits)
#
#    # dispersion parameter
#    x = layers.Dense(128, activation='relu')(targets)
#    x = layers.BatchNormalization()(x)
#    x = layers.Dense(128, activation='relu')(x)
#    x = layers.BatchNormalization()(x)
#    #x = AverageChannel()(targets)
#    #x = layers.Concatenate(axis=-1)([latent_inputs, targets])
#    #x = layers.Dense(params['nhiddendecoder'], activation='relu')(x)
#
#    p0logit = layers.Dense(1)(x)
#    r = layers.Dense(1)(x)
#    r = layers.Activation(activation=tf.math.softplus)(r)
#    r = ClipLayer(1e-10, 1e5)(r)
#
#    prob_loss = NegativeMultinomialEndpointV2()([mullogits, p0logit, r, targets])
#
#    decoder = keras.Model([latent_inputs, target_inputs],
#                           prob_loss, name="decoder")
#
#    return decoder

def create_batch_decoder(params):

    nsamples = params['nsamples']
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']
    batch_dim = params['batchnames']

    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
    
    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    batch_layer = batch_inputs

    if len(batch_layer)>1:
        batch_layer = layers.Concatenate()(batch_layer)
    else:
        batch_layer = batch_layer[0]
    batch_layer = layers.RepeatVector(nsamples)(batch_layer)

    x = layers.Concatenate()([latent_inputs, batch_layer])

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

    decoder = keras.Model([latent_inputs, target_inputs, batch_inputs],
                           prob_loss, name="decoder")

    return decoder


#def create_batch_decoder_v20(params):
#    warnings.warn("create_batch_decoder_v20 is experimental and may be removed.",
#                  type=DeprecationWarning)
#
#    nsamples = params['nsamples']
#    input_shape = (params['datadims'],)
#    latent_dim = params['latentdims']
#    batch_dim = params['batchnames']
#
#    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
#    
#    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
#    batch_layer = batch_inputs
#
#    if len(batch_layer)>1:
#        batch_layer = layers.Concatenate()(batch_layer)
#    else:
#        batch_layer = batch_layer[0]
#    batch_layer = layers.RepeatVector(nsamples)(batch_layer)
#
#    #x = layers.Concatenate()([latent_inputs, batch_layer])
#    x = latent_inputs
#
#    for nhidden in range(params['nlayers_d']):
#        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
#        x = layers.Dropout(params['hidden_d_dropout'])(x)
#
#    target_inputs = keras.Input(shape=input_shape, name='targets')
#
#    targets = layers.Reshape((1, params['datadims']))(target_inputs)
#
#    # multinomial part
#    logits = layers.Dense(params['datadims'],
#                          activation='linear', name='logits',
#                          use_bias=False)(x)
#
#    logits = AddBiasLayer(name='extra_bias')(logits)
#
#    logits_batch = layers.Dense(params['datadims'],
#                          activation='linear', name='logits_batch',
#                          use_bias=False)(batch_layer)
#
#    logits = layers.Add()([logits, logits_batch])
#
#    # dispersion parameter
#    r = ScalarBiasLayer()(x)
#    r = layers.Activation(activation=tf.math.softplus)(r)
#    r = ClipLayer(1e-10, 1e5)(r)
#
#    prob_loss = NegativeMultinomialEndpoint()([logits, r, targets])
#
#    decoder = keras.Model([latent_inputs, target_inputs, batch_inputs],
#                           prob_loss, name="decoder")
#
#    return decoder
#
#def create_batch_decoder_v21(params):
#    warnings.warn("create_batch_decoder_v21 is experimental and may be removed.",
#                  type=DeprecationWarning)
#
#    nsamples = params['nsamples']
#    input_shape = (params['datadims'],)
#    latent_dim = params['latentdims']
#    batch_dim = params['batchnames']
#
#    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
#    
#    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input') for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
#    batch_layer = batch_inputs
#
#    if len(batch_layer)>1:
#        batch_layer = layers.Concatenate()(batch_layer)
#    else:
#        batch_layer = batch_layer[0]
#    batch_layer = layers.RepeatVector(nsamples)(batch_layer)
#
#    x = layers.Concatenate()([latent_inputs, batch_layer])
#    #x = latent_inputs
#
#    for nhidden in range(params['nlayers_d']):
#        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
#        x = layers.Dropout(params['hidden_d_dropout'])(x)
#
#    target_inputs = keras.Input(shape=input_shape, name='targets')
#
#    targets = layers.Reshape((1, params['datadims']))(target_inputs)
#
#    # multinomial part
#    logits = layers.Dense(params['datadims'],
#                          activation='linear', name='logits',
#                          use_bias=False)(x)
#
#    logits = AddBiasLayer(name='extra_bias')(logits)
#
#    logits_batch = layers.Dense(params['datadims'],
#                          activation='linear', name='logits',
#                          use_bias=False)(batch_layer)
#
#    logits = layers.Add()([logits, logits_batch])
#
#    # dispersion parameter
#    r = ScalarBiasLayer()(x)
#    r = layers.Activation(activation=tf.math.softplus)(r)
#    r = ClipLayer(1e-10, 1e5)(r)
#
#    prob_loss = NegativeMultinomialEndpoint()([logits, r, targets])
#
#    decoder = keras.Model([latent_inputs, target_inputs, batch_inputs],
#                           prob_loss, name="decoder")
#
#    return decoder

def create_batchlatent_decoder(params):
    warnings.warn("create_batchlatent_decoder is experimental and may be removed.",
                  type=DeprecationWarning)
    nsamples = params['nsamples']
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']
    batch_dim = params['batchnames']

    latent_inputs = keras.Input(shape=(nsamples, latent_dim,), name='latent_input')
    
    batch_latent = keras.Input(shape=(params['nlasthiddenbatcher'],), name='batch_latent')

    batch_layer = layers.RepeatVector(nsamples)(batch_latent)
    #print(batch_layer)


    x = layers.Concatenate()([latent_inputs, batch_layer])
    #print(x)

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

    decoder = keras.Model([latent_inputs, target_inputs, batch_latent],
                           prob_loss, name="decoder")
    decoder.summary()

    return decoder

