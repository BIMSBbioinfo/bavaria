import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import numpy as np
from nmvae.vae_models import *
from nmvae.model_components import *
from nmvae.data import to_dataset
from nmvae.data import to_sparse
from scipy.stats import iqr

class EnsembleVAE:
    def __init__(self, params, repeats, output, overwrite, name='vae', feature_fraction=1.):
        self.repeats = repeats
        self.output = output
        self.models = []
        self.joined_model = None
        self.overwrite = overwrite
        if os.path.exists(output) and overwrite:
            shutil.rmtree(output)
        
        os.makedirs(output, exist_ok=True)
        self.space = params
        self.feature_fraction = max(min(feature_fraction, 1.), 0.)
        self.name = name
        print(f'using {self.name}')
        
    def _get_subfeatureset(self, X, Xt, repeat):
        """ Subset the features to speedup analysis."""
        if self.feature_fraction >= 1.:
            return X

        x_data, _ = train_test_split(Xt,
                                     train_size=self.feature_fraction,
                                     random_state=repeat*10)
        x_data = x_data.T.tocsr()
        return x_data
        
    def _get_train_test_data(self, x_data, adata, validation_split):
        """ Split training - validation input data """
        x_train, x_test = train_test_split(x_data,
                                           test_size=validation_split,
                                           random_state=42)
        return x_train, x_test

    def _get_train_test_label(self, x_data, adata, validation_split):
        """ Split training - validation labels.
        Only relevant if training with batch labels.
        """
        return None, None

    def _get_train_test(self, x_data, adata, validation_split):
        """ Split training - validation (data + labels)"""
        x_train, x_test = self._get_train_test_data(x_data, adata, validation_split)
        label_train, label_test = self._get_train_test_label(x_data, adata, validation_split)
        return x_train, x_test, label_train, label_test

    def _get_predict_label(self, adata, dummy_labels=True):
        """ get predict labels
        Only relevant if training with batch labels.
        """
        return None

    def _get_predict_data(self, x_data, adata, dummy_labels=True):
        """ get predict data + labels.
        Labels are only relevant for batch annotation.
        """
        labels = self._get_predict_label(adata, dummy_labels=dummy_labels)
        return x_data, labels
        
    def _create(self, name, space):
        """ Create VAE model"""
        if name == 'vae':
            model = VAE.create(space, create_encoder, create_decoder)
        elif name == 'bcvae':
            model = BCVAE2.create(space, create_batch_encoder, create_batch_decoder)
        elif name == 'bcvae2':
            model = BCVAE2.create(space, create_batch_encoder_alllayers, create_batch_decoder)
        elif name == 'bavaria':
            model = BAVARIA.create(space, create_batch_encoder_gan, create_batch_decoder)
        elif name == 'bavaria2':
            model = BAVARIA2.create(space, create_batch_encoder_gan, create_batchlatent_decoder, create_batchlatent_predictor)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model

    def _load(self, path):
        """ Reload VAE model"""
        if self.name == 'vae':
            model = VAE.load(path)
        elif self.name == 'bcvae':
            model = BCVAE2.load(path)
        elif self.name == 'bcvae2':
            model = BCVAE2.load(path)
        elif self.name == 'bavaria':
            model = BAVARIA.load(path)
        elif self.name == 'bavaria2':
            model = BAVARIA2.load(path)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model

    def fit(self, adata,
            shuffle=True, batch_size=64,
            epochs=1, validation_split=.15
           ):
        """ fit ensemble of VAE models (or reload pre-existing model)."""

        space = self.space
        x_data = adata.X
        x_data_t = x_data.T.tocsr()
       
        for r in range(self.repeats):
            # random feature subset
            x_subdata = self._get_subfeatureset(x_data, x_data_t, r)

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

    def _get_dataset_truebatchlabels(self, x_data, adata, batch_size=64):
        """ used without dummy labels"""
        x_data, labels = self._get_predict_data(x_data, adata, dummy_labels=False)
        tf_x = to_dataset(to_sparse(x_data), labels, shuffle=False, batch_size=batch_size)
        return tf_x

    def _get_dataset_dummybatchlabels(self, x_data, adata, batch_size=64):
        """ used with dummy labels"""
        x_data, labels = self._get_predict_data(x_data, adata, dummy_labels=True)
        tf_x = to_dataset(to_sparse(x_data), labels, shuffle=False, batch_size=batch_size)
        return tf_x

    def encode_subset(self, adata, batch_size=64):

        dfs = []
        x_data = adata.X
        x_data_t = x_data.T.tocsr()
        for i, model in enumerate(self.models):
            x_subdata = self._get_subfeatureset(x_data, x_data_t, i)

            tf_x = self._get_dataset_dummybatchlabels(x_subdata, adata, batch_size=batch_size)
 
            out = model.encoder_predict.predict(tf_x)
            df = pd.DataFrame(out, index=adata.obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            adata.obsm[f'nmvae-run_{i+1}'] = out
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        adata.obsm['nmvae-ensemble'] = df.values
        df.to_csv(os.path.join(self.output, 'latent.csv'))
        return adata


    def encode_full(self, adata, batch_size=64, skip_outliers=True):
        tf_x = self._get_dataset_truebatchlabels(adata.X, adata, batch_size=batch_size)

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
        tf_x = self._get_dataset_dummybatchlabels(adata.X, adata, batch_size=batch_size)
        for i, model in enumerate(self.models):
            if performance[i] > max_loss:
                # skip outlie
                continue
            out = model.encoder_predict.predict(tf_x)
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

class BatchEnsembleVAE(EnsembleVAE):
    def __init__(self, name, params, repeats, output, overwrite, feature_fraction=1., batchnames=[],
                 adversarial=True, conditional=False, ):
        super().__init__(params=params,
                         repeats=repeats,
                         output=output,
                         overwrite=overwrite,
                         name=name,
                         feature_fraction=feature_fraction)
        self.batchnames = batchnames
        #self.name = name
        self.adversarial = True if 'bavaria' in name else False
        self.conditional = True if 'bcvae' in name else False
        print(f'using {self.name}')
        
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
        if self.adversarial and dummy_labels:
            return None

        labels = [np.zeros_like(adata.obsm[label]) for label in self.batchnames]
        if not dummy_labels:
            return labels
        for i,_ in enumerate(labels):
            labels[i][:,0]=1
        return labels

    def _get_dataset_truebatchlabels(self, x_data, adata, batch_size=64):
        """ used without dummy labels"""
        tf_x = super()._get_dataset_truebatchlabels(x_data, adata, batch_size=batch_size)
        if self.conditional:
            tf_x = tf.data.Dataset.zip((tf_x,))
        return tf_x

    def _get_dataset_dummybatchlabels(self, x_data, adata, batch_size=64):
        """ used with dummy labels"""
        tf_x = super()._get_dataset_dummybatchlabels(x_data, adata, batch_size=batch_size)
        if self.conditional:
            tf_x = tf.data.Dataset.zip((tf_x,))
        return tf_x


