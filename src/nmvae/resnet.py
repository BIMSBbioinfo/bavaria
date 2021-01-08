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
from sklearn.preprocessing import OneHotEncoder
from anndata import AnnData
from anndata import read_h5ad
from nmvae.utils import to_dataset, to_sparse
from nmvae.utils import VAE, BCVAE
from nmvae.utils import BCVAE2
from nmvae.utils import BCVAE3 as BAVAE
from nmvae.utils import ClipLayer
from nmvae.utils import KLlossLayer
from nmvae.utils import Sampling
from nmvae.utils import ExpandDims
from nmvae.utils import BatchLoss
from nmvae.utils import ScalarBiasLayer
from nmvae.utils import AddBiasLayer
from nmvae.utils import NegativeMultinomialEndpoint
from nmvae.countmatrix import CountMatrix



def load_batch_labels(adata, batches):
    if batches is None:
        df = pd.DataFrame({'dummybatch':['dummy']*len(barcodes)},
                           index=adata.obs.index)
    elif isinstance(batches, str) and os.path.exists(batches):
        df = pd.read_csv(batches, sep='\t', index_col=0)
    return df
    
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

def one_hot_encode_batches(adata, batchnames):
    for label in batchnames:
        if label not in adata.obsm:
            oh= OneHotEncoder(sparse=False).fit_transform(adata.obs[label].values.astype(str).reshape(-1,1).tolist())
            adata.obsm[label] = oh
        if label not in adata.obs.columns:
            adata.obs.loc[:,label] = adata.obs[label].values
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
    adata = AnnData(x_data, 
                    obs=pd.DataFrame(index=cmat.cannot.barcode),
                    var=cmat.regions)
    adata.obs_names_make_unique()
    return adata


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
        ('latentdims', args.nlatent),
      ]
    )

    return params



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


def create_batch_encoder(params):
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


def create_batch_encoder_gan(params):
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

def create_batcher(params):

    nsamples = params['nsamples']
    latent_dim = params['latentdims']

    latent_input = keras.Input(shape=(nsamples, latent_dim,), name='input_batcher')

    targets = create_batch_net(latent_input, params, '')
    model = keras.Model(latent_input, targets, name='batcher')

    return model

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

class EnsembleVAE:
    def __init__(self, params, repeats, output, overwrite, feature_fraction=1.):
        self.repeats = repeats
        self.output = output
        self.models = []
        self.joined_model = None
        self.overwrite = overwrite
        if os.path.exists(output) and overwrite:
            shutil.rmtree(output)
        
        os.makedirs(output, exist_ok=True)
        self.space = params
        self.feature_fraction = feature_fraction
        self.name = 'VAE'
        
    def _get_subfeatureset(self, X, Xt, random_state):
        if self.feature_fraction < 1.:
           x_data, _ = train_test_split(Xt,
                                        train_size=self.feature_fraction,
                                        random_state=random_state)
           x_data = x_data.T.tocsr()
        else:
           x_data = X
        return x_data
        
    def _get_train_test_data(self, x_data, adata, validation_split):
        x_train, x_test = train_test_split(x_data,
                                           test_size=validation_split,
                                           random_state=42)
        return x_train, x_test

    def _get_train_test_label(self, x_data, adata, validation_split):
        return None, None

    def _get_train_test(self, x_data, adata, validation_split):
        x_train, x_test = self._get_train_test_data(x_data, adata, validation_split)
        label_train, label_test = self._get_train_test_label(x_data, adata, validation_split)
        return x_train, x_test, label_train, label_test

    def _get_predict_label(self, adata, dummy_labels=True):
        return None

    def _get_predict_data(self, x_data, adata, dummy_labels=True):
        labels = self._get_predict_label(adata, dummy_labels=dummy_labels)
        return x_data, labels
        
    def _create(self, name, space):
        if name == 'VAE':
            model = VAE.create(space, create_encoder, create_decoder)
        elif name == 'BCVAE':
            model = BCVAE2.create(space, create_batch_encoder, create_batch_decoder)
        elif name == 'BAVAE':
            model = BAVAE.create(space, create_batch_encoder_gan, create_batch_decoder)
        else:
            raise ValueError(f"Unknown model: {name}")
        #model.#summary()
        return model

    def _load(self, path):
        if self.name == 'VAE':
            model = VAE.load(path)
        elif self.name == 'BCVAE':
            model = BCVAE2.create(path)
        elif self.name == 'BAVAE':
            model = BAVAE.create(path)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model

    def fit(self, adata,
            shuffle=True, batch_size=64,
            epochs=1, validation_split=.15
           ):

        space = self.space
        x_data = adata.X
        x_data_t = x_data.T.tocsr()
       
        for r in range(self.repeats):
            # random feature subset
            x_subdata = self._get_subfeatureset(x_data, x_data_t, r*10)

            x_train, x_test, label_train, label_test = self._get_train_test(x_subdata, adata, validation_split)

            tf_X = to_dataset(to_sparse(x_train), label_train, shuffle=shuffle, batch_size=batch_size)
            tf_X_test = to_dataset(to_sparse(x_test), label_test, shuffle=False)

            output_bias = np.asarray(x_train.sum(0)).flatten()
            output_bias /= output_bias.sum()
            output_bias = np.log(output_bias + 1e-8)
        
            subpath = os.path.join(self.output, f'repeat_{r+1}')
            os.makedirs(subpath, exist_ok=True)
            if not os.path.exists(os.path.join(subpath, 'model')):

                print(f'Run repetition {r+1}')
                space['datadims'] = x_train.shape[1]
                #model = VAE.create(space, create_encoder, create_decoder)
                model = self._create(self.name, space)

                # initialize the output bias based on the overall read coverage
                # this slightly improves results
                model.decoder.get_layer('extra_bias').set_weights([output_bias])

                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True),
                             )
                csvcb = CSVLogger(os.path.join(subpath, 'train_summary.csv'))

                model.fit(tf_X, epochs = epochs,
                          validation_data=(tf_X_test,),
                          callbacks=[csvcb])
                model.save(os.path.join(subpath, 'model', 'vae.h5'))
                self.models.append(model)
            else:
                model = self._load(os.path.join(subpath, 'model', 'vae.h5'))
                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True)
                             )
                self.models.append(model)

    def determine_outliers_threshold(self, data, batch_size=64):
        tf_x_data = to_dataset(to_sparse(data), shuffle=False)
        performance = []
        for _, model in enumerate(models):
            perf = model.evaluate(tf_x_data, return_dict=True)
            performance.append(perf['loss'])

        performance = np.asarray(performance)
        max_loss = np.quantile(performance, .75) + 1.5* iqr(performance)
        self.max_loss = max_loss

    def get_models(self):
        if len(self.models) < self.repeats:
            for r in range(self.repeats):
                model = VAE.load(os.path.join(self.output, f'repeat_{r+1}', 'model.h5'))
                self.models.append(model)

    def encode_subset(self, adata, batch_size=64):

        dfs = []
        data = adata.X
        x_data_t = data.T.tocsr()
        for i, model in enumerate(self.models):
            x_subdata = self._get_subfeatureset(x_data, x_data_t, i*10)
            x_subdata, labels = self._get_predict_data(x_subdata, adata)

            tf_x = to_dataset(to_sparse(x_subdata), labels, shuffle=False, batch_size=batch_size)

            predmodel = keras.Model(model.encoder.inputs, model.encoder.get_layer('z_mean').output)
            out = predmodel.predict(tf_x)
            df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            adata.obsm[f'nmvae-run_{i+1}'] = out
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        adata.obsm['nmvae-ensemble'] = df.values
        df.to_csv(os.path.join(self.output, 'latent.csv'))
        return adata


    def encode_full(self, adata, batch_size=64, skip_outliers=True):
        x_subdata, labels = self._get_predict_data(adata.X, adata, dummy_labels=False)
        tf_x = to_dataset(to_sparse(x_subdata), labels, shuffle=False, batch_size=batch_size)

        performance = []
        for model in self.models:
            perf = model.evaluate(tf_x, return_dict=True)
            performance.append(perf['loss'])

        performance = np.asarray(performance)
        if skip_outliers:
            max_loss = np.quantile(performance, .75) + 1.5* iqr(performance)
        else:
            max_loss = max(performance)

        dfs = []
        x_subdata, labels = self._get_predict_data(adata.X, adata, dummy_labels=True)
        tf_x = to_dataset(to_sparse(x_subdata), labels, shuffle=False, batch_size=batch_size)
        for i, model in enumerate(self.models):
            if performance[i] > max_loss:
                # skip outlie
                continue
            predmodel = keras.Model(model.encoder.inputs, model.encoder.get_layer('z_mean').output)
            out = predmodel.predict(tf_x)
            df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            adata.obsm[f'nmvae-run_{i+1}'] = out
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        adata.obsm['nmvae-ensemble'] = df.values
        df.to_csv(os.path.join(self.output, 'latent.csv'))
        return adata

    def encode(self, adata, batch_size=64):
        if self.feature_fraction < 1.:
            return self.encode_subset(adata, batch_size)
        else:
            return self.encode_full(adata, batch_size)

class BatchAdversarialEnsembleVAE(EnsembleVAE):

    def _get_train_test_label(self, x_data, adata, validation_split):
        labels_train=[]
        labels_test=[]
        for label  in self.batchnames:
            label_train, label_test = train_test_split(adata.obsm[label],
                                               test_size=validation_split,
                                               random_state=42)
            labels_train.append(label_train)
            labels_test.append(label_test)
        return labels_train, labels_test
                
    def __init__(self, params, repeats, output, overwrite, feature_fraction=1., batchnames=[]):
        super().__init__(params=params,
                         repeats=repeats,
                         output=output,
                         overwrite=overwrite,
                         feature_fraction=feature_fraction)
        self.batchnames = batchnames
        self.name = 'BAVAE'
        
    def _get_predict_label(self, adata, dummy_labels=True):
        labels = [np.zeros_like(adata.obsm[label]) for label in self.batchnames]
        if not dummy_labels:
            return labels
        for i,_ in enumerate(labels):
            labels[i][:,0]=1
        return labels

    def encode_full(self, adata, batch_size=64, skip_outliers=True):
        x_subdata, labels = self._get_predict_data(adata.X, adata, dummy_labels=False)
        tf_x = to_dataset(to_sparse(x_subdata), labels, shuffle=False, batch_size=batch_size)

        performance = []
        for model in self.models:
            perf = model.evaluate(tf_x, return_dict=True)
            performance.append(perf['loss'])

        performance = np.asarray(performance)
        if skip_outliers:
            max_loss = np.quantile(performance, .75) + 1.5* iqr(performance)
        else:
            max_loss = max(performance)

        dfs = []
        #x_subdata, _ = self._get_predict_data(adata.X, adata, dummy_labels=True)
        tf_x = to_dataset(to_sparse(x_subdata), None, shuffle=False, batch_size=batch_size)
        for i, model in enumerate(self.models):
            if performance[i] > max_loss:
                # skip outlie
                continue
            #predmodel = keras.Model(model.encoder.inputs, model.encoder.get_layer('z_mean').output)
            #out = predmodel.predict(tf_x)
            out = model.encoder_predict(tf_x)
            df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            adata.obsm[f'nmvae-run_{i+1}'] = out
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        adata.obsm['nmvae-ensemble'] = df.values
        df.to_csv(os.path.join(self.output, 'latent.csv'))
        return adata

    #def encode_subset(self, adata, batch_size=64):

    #    dfs = []
    #    data = adata.X
    #    x_data_t = data.T.tocsr()
    #    for i, model in enumerate(self.models):
    #        x_subdata = self._get_subfeatureset(x_data, x_data_t, r*10)
    #        x_subdata, labels = self._get_predict_data(x_subdata, adata)

    #        tf_x = to_dataset(to_sparse(x_subdata), labels, shuffle=False, batch_size=batch_size)

    #        predmodel = keras.Model(model.encoder.inputs, model.encoder.get_layer('z_mean').output)
    #        out = predmodel.predict(tf_x)
    #        df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
    #        df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
    #        adata.obsm[f'nmvae-run_{i+1}'] = out
    #        dfs.append(df)
    #    df = pd.concat(dfs, axis=1)
    #    adata.obsm['nmvae-ensemble'] = df.values
    #    df.to_csv(os.path.join(self.output, 'latent.csv'))
    #    return adata


class BatchConditionalEnsembleVAE(EnsembleVAE):

    def _get_train_test_label(self, x_data, adata, validation_split):
        labels_train=[]
        labels_test=[]
        for label  in self.batchnames:
            label_train, label_test = train_test_split(adata.obsm[label],
                                               test_size=validation_split,
                                               random_state=42)
            labels_train.append(label_train)
            labels_test.append(label_test)
        return labels_train, labels_test
                
    def _get_predict_label(self, adata, dummy_labels=True):
        labels = [np.zeros_like(adata.obsm[label]) for label in self.batchnames]
        if not dummy_labels:
            return labels
        for i,_ in enumerate(labels):
            labels[i][:,0]=1
        return labels


    def __init__(self, params, repeats, output, overwrite, feature_fraction=1., batchnames=[]):
        super().__init__(params=params,
                         repeats=repeats,
                         output=output,
                         overwrite=overwrite,
                         feature_fraction=feature_fraction)
        self.batchnames = batchnames
        self.name = 'BCVAE'
        
    def encode_full(self, adata, batch_size=64, skip_outliers=True):
        data, labels = self._get_predict_data(adata.X, adata, dummy_labels=False)
        tf_x = to_dataset(to_sparse(data), labels, shuffle=False, batch_size=batch_size)

        performance = []
        for model in self.models:
            perf = model.evaluate(tf_x, return_dict=True)
            performance.append(perf['loss'])

        performance = np.asarray(performance)
        if skip_outliers:
            max_loss = np.quantile(performance, .75) + 1.5* iqr(performance)
        else:
            max_loss = max(performance)

        dfs = []
        data, labels = self._get_predict_data(adata.X, adata, dummy_labels=True)
        tf_x = to_dataset(to_sparse(data), labels, shuffle=False, batch_size=batch_size)
        for i, model in enumerate(self.models):
            if performance[i] > max_loss:
                # skip outlie
                continue
            predmodel = model.encoder_predict
            out = np.concatenate([predmodel(inp, training=False) for inp in tf_x], axis=0)
            df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            adata.obsm[f'nmvae-run_{i+1}'] = out
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        adata.obsm['nmvae-ensemble'] = df.values
        df.to_csv(os.path.join(self.output, 'latent.csv'))
        return adata

    #def encode_subset(self, adata, batch_size=64):

    #    dfs = []
    #    data = adata.X
    #    x_data_t = data.T.tocsr()
    #    for i, model in enumerate(self.models):
    #        x_subdata = self._get_subfeatureset(x_data, x_data_t, i*10)
    #        x_subdata, labels = self._get_predict_data(x_subdata, adata)

    #        tf_x = to_dataset(to_sparse(x_subdata), labels, shuffle=False, batch_size=batch_size)

    #        predmodel = keras.Model(model.encoder.inputs, model.encoder.get_layer('z_mean').output)
    #        out = predmodel.predict(tf_x)
    #        df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
    #        df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
    #        adata.obsm[f'nmvae-run_{i+1}'] = out
    #        dfs.append(df)
    #    df = pd.concat(dfs, axis=1)
    #    adata.obsm['nmvae-ensemble'] = df.values
    #    df.to_csv(os.path.join(self.output, 'latent.csv'))
    #    return adata
