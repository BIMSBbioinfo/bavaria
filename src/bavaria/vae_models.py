import os
import tensorflow as tf
from bavaria.layers import Sampling
from bavaria.layers import KLlossLayer
from bavaria.layers import ClipLayer
from bavaria.layers import NegativeMultinomialEndpoint
from bavaria.layers import NegativeMultinomialEndpointV2
from bavaria.layers import AddBiasLayer
from bavaria.layers import ScalarBiasLayer
from bavaria.layers import BatchLoss
from bavaria.layers import ExpandDims
from bavaria.layers import MutInfoLayer
from keras.models import load_model

class VAE(tf.keras.Model):
    """
    z = encoder(data)
    recon = decoder([z, data])
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = tf.keras.Model(self.encoder.inputs,
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
                          'NegativeMultinomialEndpointV2': NegativeMultinomialEndpointV2,
                          'AddBiasLayer': AddBiasLayer,
                          'MutInfoLayer': MutInfoLayer,
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

class BCVAE(tf.keras.Model):
    """
    z, batch_pred = encoder(data)
    recon = decoder([z, data, batch])
    """
    def __init__(self, encoder, decoder, batcher, **kwargs):
        super(BCVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = tf.keras.Model(self.encoder.inputs,
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
            data, batch = data
        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder(data)
            for i, loss in enumerate(self.encoder.losses):
                losses[f'kl_loss_{i}'] = loss
            pred = self.decoder([z, data, batch])
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            batchpred = self.batcher(z)
            if not isinstance(batchpred, tuple):
                batchpred = (batchpred,)
            batch_loss = []
            for i, (pred, labs) in enumerate(zip(batchpred, batch)):
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
            data, batch = data
        z = self.encoder(data)
        pred = self.decoder([z, data, batch])
        batchpred = self.batcher(z)

        if not isinstance(batchpred, tuple):
            batchpred = (batchpred,)
        batch_loss = []
        for i, (pred, labs) in enumerate(zip(batchpred, batch)):
            batch_loss.append(self.cce(labs, tf.math.reduce_mean(pred, axis=1)))

        batch_loss = sum(batch_loss)
        total_loss = sum(self.encoder.losses) + sum(self.decoder.losses) - batch_loss

        losses['loss'] = total_loss
        return losses


    def predict_step(self, data):
        data = data[0]
        if isinstance(data, tuple):
            data, batch = data
        z = self.encoder(data)
        pred = self.decoder([z, data, batch])
        return pred


class BCVAE2(tf.keras.Model):
    """
    z = encoder([data, batch])
    recon = decoder([z, data, batch])
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(BCVAE2, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = tf.keras.Model(self.encoder.inputs,
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
                          'MutInfoLayer': MutInfoLayer,
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
            data, batch = data
        z = self.encoder([data, batch])
        if len(self.decoder.losses) > 0:
            pred = self.decoder([z, data, batch])
        else:
            pred = self.decoder(z)

        return pred

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            data, batch = data
        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder([data, batch])
            for i, loss in enumerate(self.encoder.losses):
                losses[f'kl_loss_{i}'] = loss
            pred = self.decoder([z, data, batch])
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
            data, batch = data
        z = self.encoder([data, batch])
        pred = self.decoder([z, data, batch])

        total_loss = sum(self.encoder.losses) + sum(self.decoder.losses)

        losses['loss'] = total_loss
        return losses

    def predict_step(self, data):
        #print('predict_step', data)
        data = data[0]
        if isinstance(data, tuple):
            data, batch = data
        
        z = self.encoder([data, batch])
        pred = self.decoder([z, data, batch])
        return pred

class BAVARIA(tf.keras.Model):
    """
    z, batch_pred = encoder([data, batch])
    recon = decoder([z, data, batch])
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(BAVARIA, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = tf.keras.Model(self.encoder.get_layer('input_data').input,
                                           self.encoder.get_layer('z_mean').output)

        self.decoder = decoder

        self.encoder_params = tf.keras.Model(self.encoder.get_layer('input_data').input,
                                          self.encoder.get_layer('random_latent').output).trainable_weights
        self.batch_params = [w for w in encoder.trainable_weights if 'batchcorrect' in w.name]

        ba = [l.output for l in self.encoder.layers if 'combine_batches' in l.name]
        test_encoder = tf.keras.Model(self.encoder.get_layer('input_data').input,
                                   [self.encoder.get_layer('random_latent').output, ba], name="encoder")
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
                          'ExpandDims': ExpandDims,
                          'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                          'AddBiasLayer': AddBiasLayer,
                          'MutInfoLayer': MutInfoLayer,
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
            data, batch = data
        z = self.encoder([data, batch])
        if len(self.decoder.losses) > 0:
            pred = self.decoder([z, data, batch])
        else:
            pred = self.decoder(z)

        return pred

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            profile, batch = data

        with tf.GradientTape(persistent=True) as tape:
            z, b = self.encoder([profile, batch])
            kl_loss, batch_loss = self.encoder.losses
            losses['kl_loss'] = kl_loss
            losses['bloss'] = batch_loss

            pred = self.decoder([z, profile, batch])
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
            profile, batch = data
        z, b = self.encoder([profile, batch])

        kl_loss, batch_loss = self.encoder.losses
        pred = self.decoder([z, profile, batch])
        recon_loss = sum(self.decoder.losses)

        total_loss = kl_loss + recon_loss - batch_loss

        losses['loss'] = total_loss
        for mn, s in zip(self.encoder.metrics_names, self.encoder.metrics):
            losses[mn] = s.result()
        return losses


class BAVARIA2(tf.keras.Model):
    """ Experimental BAVARIA with batch-feature extractor."""
    def __init__(self, encoder, decoder, batch_predictor, **kwargs):
        super(BAVARIA2, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = tf.keras.Model(self.encoder.get_layer('input_data').input,
                                           self.encoder.get_layer('z_mean').output)

        self.decoder = decoder

        self.encoder_params = tf.keras.Model(self.encoder.get_layer('input_data').input,
                                          self.encoder.get_layer('random_latent').output).trainable_weights
        self.batch_params = [w for w in encoder.trainable_weights if 'batchcorrect' in w.name]

        self.batch_predictor = batch_predictor

    def save(self, filename):
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'
        self.encoder.save(f + '_encoder_' + s)
        self.decoder.save(f + '_decoder_' + s)
        self.batch_predictor.save(f + '_batch_' + s)

    @classmethod
    def create(cls, params, _create_encoder, _create_decoder, _create_batch_predictor):
         encoder = _create_encoder(params)
         decoder = _create_decoder(params)
         batch_predictor = _create_batch_predictor(params)

         return cls(encoder, decoder, batch_predictor)

    @classmethod
    def load(cls, filename):
        f = filename.split('.h5')[0]
        s='.h5'
        
        custom_objects = {'Sampling': Sampling,
                          'KLlossLayer': KLlossLayer,
                          'ClipLayer': ClipLayer,
                          'BatchLoss': BatchLoss,
                          'ExpandDims': ExpandDims,
                          'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                          'AddBiasLayer': AddBiasLayer,
                          'MutInfoLayer': MutInfoLayer,
                          'ScalarBiasLayer':ScalarBiasLayer,
                         }
        encoder = load_model(f + '_encoder_' + s, custom_objects=custom_objects)
        decoder = load_model(f + '_decoder_' + s, custom_objects=custom_objects)
        batch_predictor = load_model(f + '_batch_' + s, custom_objects=custom_objects)
        return cls(encoder, decoder, batch_predictor)

    def save_weights(self, filename, overwrite=True, save_format=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.save_weights(f + '_encoder_' + s)
        self.decoder.save_weights(f + '_decoder_' + s)
        self.batch_predictor.save_weights(f + '_batch_' + s)

    def load_weights(self, filename, by_name=False, skip_mismatch=False):
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.load_weights(f + '_encoder_' + s)
        self.decoder.load_weights(f + '_decoder_' + s)
        self.batch_predictor.load_weights(f + '_batch_' + s)

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
        self.batch_predictor.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            profile, labels = data

        with tf.GradientTape(persistent=True) as tape:
            z, b = self.encoder([profile, labels])
            kl_loss, batch_loss = self.encoder.losses
            losses['kl_loss'] = kl_loss
            losses['bloss'] = batch_loss

            latent, pred = self.batch_predictor([profile, labels])
            predict_loss = sum(self.batch_predictor.losses)
            losses[f'batch_loss'] = predict_loss

            pred = self.decoder([z, profile, latent])
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            recon_loss = sum(self.decoder.losses)
            total_loss = kl_loss + recon_loss - batch_loss

        grads = tape.gradient(predict_loss, self.batch_predictor.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.batch_predictor.trainable_weights))

        grads = tape.gradient(total_loss, self.encoder_params + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder_params + self.decoder.trainable_weights))

        grads = tape.gradient(batch_loss, self.batch_params)
        self.optimizer.apply_gradients(zip(grads, self.batch_params))

        del tape
        losses['loss'] = total_loss
        for mn, s in zip(self.encoder.metrics_names, self.encoder.metrics):
            losses[mn] = s.result()
        for mn, s in zip(self.batch_predictor.metrics_names, self.batch_predictor.metrics):
            losses['bp'+mn] = s.result()
        return losses


    def test_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            profile, labels = data
        z, b = self.encoder(data)

        kl_loss, batch_loss = self.encoder.losses

        latent, pred = self.batch_predictor([profile, labels])
        predict_loss = sum(self.batch_predictor.losses)
        losses[f'batch_loss'] = predict_loss

        pred = self.decoder([z, profile, latent])
        recon_loss = sum(self.decoder.losses)

        total_loss = kl_loss + recon_loss - batch_loss

        losses['loss'] = total_loss
        for mn, s in zip(self.encoder.metrics_names, self.encoder.metrics):
            losses[mn] = s.result()
        return losses


