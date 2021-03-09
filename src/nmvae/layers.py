import tensorflow as tf
from tensorflow.keras import layers

@tf.function
def multinomial_likelihood(targets, p):
    return tf.reduce_sum(tf.math.xlogy(targets, p+1e-10), axis=-1)


@tf.function
def negative_multinomial_likelihood(targets, logits, r):
    likeli = tf.reduce_sum(tf.math.xlogy(targets, softmax1p(logits)+1e-10), axis=-1)
    tf.debugging.check_numerics(likeli, "targets * log(p)")
    likeli += tf.reduce_sum(tf.math.xlogy(r, softmax1p0(logits) + 1e-10), axis=-1)
    tf.debugging.check_numerics(likeli, "r * log(1-p)")
    likeli += tf.math.lgamma(tf.reduce_sum(r, axis=-1) + tf.reduce_sum(targets, axis=-1))
    tf.debugging.check_numerics(likeli, "lgamma(r + x)")
    likeli -= tf.reduce_sum(tf.math.lgamma(r), axis=-1)
    tf.debugging.check_numerics(likeli, "lgamma(r)")
    return likeli


@tf.function
def negative_multinomial_likelihood_v2(targets, mul_logits, p0_logits, r):
    X = tf.reduce_sum(targets, axis=-1)
    #nb likelihood
    likeli = tf.math.lgamma(tf.reduce_sum(r, axis=-1) + X)
    likeli -= tf.reduce_sum(tf.math.lgamma(r), axis=-1)
    likeli += tf.reduce_sum(tf.math.xlogy(r, tf.math.sigmoid(p0_logits)+1e-10), axis=-1)

    # mul likelihood
    p = _softmax(mul_logits)*tf.math.sigmoid(-p0_logits)
    likeli += tf.reduce_sum(tf.math.xlogy(targets, p+1e-10), axis=-1)
    tf.debugging.check_numerics(likeli, "negative_multinomial_likelihood_v2")
    return likeli


@tf.function
def _softmax(x):
    xmax = tf.reduce_max(x, axis=-1, keepdims=True)
    x = x - xmax
    sp = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)
    tf.debugging.check_numerics(sp, "_softmax is NaN")
    return sp


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

class AverageChannel(layers.Layer):
    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis=-1, keepdims=True)
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)
    

class ScalarBiasLayer(layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(1,),
                                    initializer='ones',
                                    trainable=True)
    def call(self, x):
        return tf.ones((tf.shape(x)[0],) + (1,)*(len(x.shape.as_list())-1))*self.bias


class AddBiasLayer(layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight('extra_bias',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + tf.expand_dims(self.bias, 0)

class NegativeMultinomialEndpoint(layers.Layer):
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


class NegativeMultinomialEndpointV2(layers.Layer):
    def call(self, inputs):
        targets = None
        if len(inputs) == 4:
            mul_logits, p0_logit, r, targets = inputs
        else:
            mul_logits, p0_logit, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           negative_multinomial_likelihood_v2(targets, mul_logits, p0_logit, r)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "NegativeMultinomialEndpoint NaN")

        p0 = tf.math.sigmoid(p0_logit)
        p1 = tf.math.sigmoid(-p0_logit)

        pm = _softmax(mul_logits)
        p = pm * p1
        return p * r / (p0 + 1e-10)


