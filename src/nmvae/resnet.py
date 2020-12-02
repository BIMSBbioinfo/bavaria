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
from nmvae.utils import to_dataset, to_sparse
from nmvae.utils import VAE, BCVAE
from nmvae.utils import ClipLayer
from nmvae.utils import KLlossLayer
from nmvae.utils import Sampling
from nmvae.utils import ScalarBiasLayer
from nmvae.utils import AddBiasLayer
from nmvae.utils import NegativeMultinomialEndpoint
from nmvae.countmatrix import CountMatrix



def load_batch_labels(barcodes, batches):
    if batches is None:
        df = pd.DataFrame({'barcode':barcodes, 'dummybatch':['dummy']*len(barcodes)})
    else:
        df = pd.read_csv(batches, sep='\t')
    return df
    
def resnet_vae_batch_params(df):
    columns = df.columns[1:].values.tolist()
    ncats = [df[c].unique().shape[0] for c in columns]
    params = OrderedDict([
       ('batchnames', columns),
       ('nbatchcats', ncats),
         ])
    return params

def one_hot_encode_batches(adata, df):
    for label in df.columns[1:].values.tolist():
        adata.obsm[label] = OneHotEncoder(sparse=False).fit_transform(df[label].values.reshape(-1,1))
        adata.obs.loc[:,label] = df[label].values
    return adata

def load_data(data, regions, cells):
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


def create_encoder_nonvariational(params):
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
    #z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    #z_log_var = ClipLayer(-10., 10.)(z_log_var)

    #z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    #z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])
    z = layers.RepeatVector(params['nsamples'], name='deterministic_latent')(z_mean)

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder


def create_batch_encoder(params):
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']
    batch_dim = params['batchdim']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    batch_inputs = keras.Input(shape=(batch_dim,), name='batch_input')

    batch_layer = layers.Reshape((1, batch_dim))(batch_inputs)

    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)
    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)

    xbatchhidden = layers.Dense(params['nhidden_b'],
                                activation='relu')(batch_layer)

    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    x = xinit

    x = layers.Concatenate()([x, xbatchhidden])

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(params['nsamples'], name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder


def create_batcher(params):

    nsamples = params['nsamples']
    latent_dim = params['latentdims']

    latent_input = keras.Input(shape=(nsamples, latent_dim,), name='input_batcher')

    x = layers.Dense(params['nhiddenbatcher'], activation='relu')(latent_input)

    targets = [layers.Dense(nl, activation='softmax', name=name)(x) \
               for nl,name in zip(params['nbatchcats'], params['batchnames'])]
    model = keras.Model(latent_input, targets, name='batcher')

    return model


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

class MetaVAE:
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
        
    def fit(self, adata,
            shuffle=True, batch_size=64,
            epochs=1, validation_split=.15
           ):

        space = self.space
        x_data = adata.X
        x_data_t = x_data.T.tocsr()
       
        for r in range(self.repeats):
            # random feature subset
            if self.feature_fraction < 1.:
               x_data_ts, _ = train_test_split(x_data_t, train_size=self.feature_fraction, random_state=r*10)
               x_data_ts = x_data_ts.T.tocsr()
            else:
               x_data_ts = x_data

            
            x_train, x_test = train_test_split(x_data_ts, test_size=validation_split,
                                               random_state=42)

            
            tf_X = to_dataset(to_sparse(x_train), shuffle=shuffle, batch_size=batch_size)

            output_bias = np.asarray(x_train.sum(0)).flatten()
            output_bias /= output_bias.sum()
            output_bias = np.log(output_bias + 1e-8)
        
            subpath = os.path.join(self.output, f'repeat_{r+1}')
            os.makedirs(subpath, exist_ok=True)
            if not os.path.exists(os.path.join(subpath, 'model')):

                print(f'Run repetition {r+1}')
                space['datadims'] = x_train.shape[1]
                model = VAE.create(space, create_encoder, create_decoder)

                # initialize the output bias based on the overall read coverage
                # this slightly improves results
                model.decoder.get_layer('extra_bias').set_weights([output_bias])

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
            #model.summary()

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
            if self.feature_fraction < 1.:
               x_data_ts, _ = train_test_split(x_data_t, train_size=self.feature_fraction, random_state=i*10)
               x_data_ts = x_data_ts.T.tocsr()
            else:
               x_data_ts = data

            tf_x = to_dataset(to_sparse(x_data_ts), shuffle=False, batch_size=batch_size)

            predmodel = keras.Model(model.encoder.inputs, model.encoder.get_layer('z_mean').output)
            out = predmodel.predict(tf_x)
            df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        adata.obsm['nmvae-ensemble'] = df.values
        df.to_csv(os.path.join(self.output, 'latent.csv'))
        return adata


    def encode_full(self, adata, batch_size=64, skip_outliers=True):
        data = adata.X
        tf_x = to_dataset(to_sparse(data), shuffle=False, batch_size=batch_size)

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

    def variable_regions(self, adata, batch_size=64):

        data = adata.X

        tf_x = to_dataset(to_sparse(data),
                          shuffle=False,
                          batch_size=batch_size)

        df = pd.DataFrame(columns=['chr', 'start', 'end', 'max','min', 'mean', 'var'])

        psum = None
        pmin = None
        pmax = None
        psq = None
        n = 0
        for xinput in tf_x:
            prediction = []
            for model in self.models:
                prediction.append(model.predict(xinput))

            # r x b x s x f
            p = np.asarray(prediction)
            p = p/p.sum(-1, keepdims=True)

            n += np.prod(p.shape[:-1])

            if psum is None:
                psum = p.sum((0,1,2))
                pmin = p.min((0,1,2))
                pmax = p.max((0,1,2))
                psq = np.square(p).sum((0,1,2))
            else:
                psum += p.sum((0,1,2))
                pmin = np.minimum(pmin, p.min((0,1,2)))
                pmax = np.maximum(pmax, p.max((0,1,2)))
                psq += np.square(p).sum((0,1,2))
                
        pmean = psum / n
        var = psq / n - np.square(pmean)
        
        #df = regions.copy()
        #df.loc[:,"mean"] = pmean
        #df.loc[:,"var"] = psq / n - np.square(pmean)
        #df.loc[:,"min"] = pmin
        #df.loc[:,"max"] = pmax

        #df.to_csv(os.path.join(self.output, 'variable_regions.tsv'), sep="\t", index=False)
        adata.var.loc[:,"mean"] = pmean
        adata.var.loc[:,"var"] = var
        adata.var.loc[:,"sd"] = np.sqrt(var)
        adata.var.loc[:,"min"] = pmin
        adata.var.loc[:,"max"] = pmax
        return adata

class BatchMetaVAE(MetaVAE):

    def __init__(self, params, repeats, output, overwrite, feature_fraction=1., batchnames=[]):
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
        self.batchnames = batchnames
        
    def fit(self, adata, 
            shuffle=True, batch_size=64,
            epochs=1, validation_split=.15
           ):

        space = self.space
        x_data = adata.X
        x_data_t = x_data.T.tocsr()
       
        for r in range(self.repeats):
            # subsample features if necessary
            if self.feature_fraction < 1.:
               x_data_ts, _ = train_test_split(x_data_t,
                                               train_size=self.feature_fraction,
                                               random_state=r*10)
               x_data_ts = x_data_ts.T.tocsr()
            else:
               x_data_ts = x_data

            x_train, x_test = train_test_split(x_data_ts,
                                               test_size=validation_split,
                                               random_state=42)

            labels_train=[]
            labels_test=[]
            for label  in self.batchnames:
                label_train, label_test = train_test_split(adata.obsm[label],
                                                   test_size=validation_split,
                                                   random_state=42)
                labels_train.append(label_train)
                labels_test.append(label_test)
                
            tf_X = to_dataset(to_sparse(x_train), labels_train,
                              shuffle=shuffle, batch_size=batch_size)
            tf_X_test = to_dataset(to_sparse(x_test), labels_test, shuffle=False)

            output_bias = np.asarray(x_train.sum(0)).flatten()
            output_bias /= output_bias.sum()
            output_bias = np.log(output_bias + 1e-8)
        
            subpath = os.path.join(self.output, f'repeat_{r+1}')
            os.makedirs(subpath, exist_ok=True)
            if not os.path.exists(os.path.join(subpath, 'model')):

                print(f'Run repetition {r+1}')
                space['datadims'] = x_train.shape[1]
                model = BCVAE.create(space, create_encoder, create_batch_decoder, create_batcher)

                # initialize the output bias based on the overall read coverage
                # this slightly improves results
                model.decoder.get_layer('extra_bias').set_weights([output_bias])

                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True)
                             )
                csvcb = CSVLogger(os.path.join(subpath, 'train_summary.csv'))

                model.fit(tf_X, epochs = epochs,
                          validation_data=(tf_X_test,),
                          callbacks=[csvcb])
                model.save(os.path.join(subpath, 'model', 'vae.h5'))
                self.models.append(model)
            else:
                model = BCVAE.load(os.path.join(subpath, 'model', 'vae.h5'))
                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True)
                             )
                self.models.append(model)
            #model.summary()


    def encode_full(self, adata, batch_size=64, skip_outliers=True):

        data = adata.X
        labels = [adata.obsm[label] for label in self.batchnames]
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

        tf_x = to_dataset(to_sparse(data), None, shuffle=False, batch_size=batch_size)
        dfs = []
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

    def variable_regions(self, adata, batch_size=64):

        data = adata.X
        # define dummy batch labels such that all cells have a common batch
        labels = [np.zeros_like(adata.obsm[label]) for label in self.batchnames]
        for i,_ in enumerate(labels):
            labels[i][:,0]=1
        #labels = [adata.obsm[label]) for label in self.batchnames]

        tf_x = to_dataset(to_sparse(data), labels,
                          shuffle=False,
                          batch_size=batch_size)

        df = pd.DataFrame(columns=['chr', 'start', 'end', 'max','min', 'mean', 'var'])

        psum = None
        pmin = None
        pmax = None
        psq = None
        n = 0
        for xinput in tf_x:
            prediction = []
            for model in self.models:
                prediction.append(model.predict(xinput))

            # r x b x s x f
            p = np.asarray(prediction)
            p = p/p.sum(-1, keepdims=True)

            n += np.prod(p.shape[:-1])

            if psum is None:
                psum = p.sum((0,1,2))
                pmin = p.min((0,1,2))
                pmax = p.max((0,1,2))
                psq = np.square(p).sum((0,1,2))
            else:
                psum += p.sum((0,1,2))
                pmin = np.minimum(pmin, p.min((0,1,2)))
                pmax = np.maximum(pmax, p.max((0,1,2)))
                psq += np.square(p).sum((0,1,2))
                
        pmean = psum / n
        var = psq / n - np.square(pmean)
        
        #df = regions.copy()
        #df.loc[:,"mean"] = pmean
        #df.loc[:,"var"] = psq / n - np.square(pmean)
        #df.loc[:,"min"] = pmin
        #df.loc[:,"max"] = pmax

        #df.to_csv(os.path.join(self.output, 'variable_regions.tsv'), sep="\t", index=False)
        adata.var.loc[:,"mean"] = pmean
        adata.var.loc[:,"var"] = var
        adata.var.loc[:,"sd"] = np.sqrt(var)
        adata.var.loc[:,"min"] = pmin
        adata.var.loc[:,"max"] = pmax
        return adata

