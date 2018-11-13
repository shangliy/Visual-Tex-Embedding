"""
File: train_vte_extractor.py
Author: Shanglin Yang
Description: Train Visual-Text-Embedding Extractor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from configurations import ModelConfig
from layers import ContrastiveLoss, contrast_loss
from dataloader import PairdataLoader

from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

from keras.layers import (Dense, Embedding, GRU, Input, LSTM, RepeatVector,
                          TimeDistributed)
from keras.layers import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.regularizers import l1_l2


class VteModel(object):
    """Model for visual text embedding Extractor
    """
    def __init__(self, config, mode):
        """Basic setup.
        Args:
            config: Object containing configuration parameters.
            mode: "train", "eval" or "inference".
            train_inception: Whether the inception submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode

        self._regularizer = l1_l2(config.l1, config.l2)
        self._initializer = 'random_uniform'

        # data loader for pair image and text
        self.dataloader = PairdataLoader(self.config)
        self._build_model()

    def _build_image_embedding(self):
        image_model = InceptionV3(include_top=False, weights='imagenet',
                                  pooling='avg')
        for layer in image_model.layers:
            layer.trainable = False
        
        dense_input = BatchNormalization(axis=-1)(image_model.output)
        image_embedding = Dense(units=self.config._embedding_size,
                            kernel_regularizer=self._regularizer,
                            kernel_initializer=self._initializer
                            )(dense_input)
        #image_embedding = RepeatVector(1)(image_dense)
        image_input = image_model.input
        return image_input, image_embedding
    
    def _build_word_embedding(self):

        sentence_input = Input(shape=[None])
        word_embedding_seq = Embedding(
                                    input_dim=self.config._vocab_size + 1,
                                    output_dim=self.config._embedding_size,
                                    embeddings_regularizer=self._regularizer
                                    )(sentence_input)
        word_embedding = GlobalAveragePooling1D()(word_embedding_seq)
        return sentence_input, word_embedding
    
    def _build_model(self):
        """Build the model.
        """
        
        self.image_input, self.image_embedding = self._build_image_embedding()
        self.sentence_input, self.words_embedding = self._build_word_embedding()
        self.emb_margin = Input(shape=[1,])

        self.emb_batch_loss = ContrastiveLoss()([self.words_embedding, self.image_embedding, self.emb_margin])

        self.model = Model([self.image_input, self.sentence_input, self.emb_margin],
                            self.emb_batch_loss)
        
        self.model.compile(optimizer='SGD', loss=contrast_loss)

    
    def _fit(self, log=False, **kwargs):

        if log:
            callbacks = [keras.callbacks.TensorBoard(
                log_dir='./log', histogram_freq=0,
                write_graph=True, write_images=False)]
        else:
            callbacks = []
        
        gen = self.dataloader._batch_generator_train(self.config.batch_size)

        steps_per_epoch = self.dataloader.sample_number//self.config.batch_size

        self.model.fit_generator(
            gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.epochs,
            shuffle=True,
            callbacks=callbacks,
            pickle_safe=True,
            **kwargs)
    
    def _eval(self):

        gen = self.dataloader._batch_generator_val(self.config.batch_size)
        [imgs, txts, margin], y = next(gen)

        con_loss = self.model.predict([imgs, txts, margin])
        
        print("Contrast loss:", con_loss)





if __name__ == '__main__':
    # load config
    config = ModelConfig()

    model = VteModel(config, 'train')
    print(model.model.output)
    #model._fit(log=False)
    model._eval()


