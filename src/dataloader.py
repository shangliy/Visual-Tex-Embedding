# This is the data loader for JD data
import math
import numpy as np
import pickle as pkl
from keras.preprocessing import image
from keras.preprocessing import sequence

from keras.applications.inception_v3 import preprocess_input


class PairdataLoader(object):
    def __init__(self, config):
        
        self.margin = config.emb_margin
        self.raw_data = pkl.load(open(config.data_path,'rb'))
        self.vocab_list = pkl.load(open(config.vocab_path, 'rb'))

        self.img_size = config.img_size
        self.max_len = config.max_len
        self.tv_ratio = config.tv_ratio

        self.sample_number = len(self.raw_data)
        self.train_image_list, self.train_texts_list = [], []
        self.val_image_list, self.val_texts_list = [], []

        self.get_pairlist()

        
    def _image_preprocessing(self, image_path):
        img = image.load_img(image_path, target_size=(self.img_size, self.img_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        return x
    
    def _text_preprocessing(self, seq):
        word_list = seq.split()
        s = [self.vocab_list.index(w) + 1 for w in word_list]
        s = sequence.pad_sequences([s], maxlen=self.max_len)
        return s
    
    def get_pairlist(self):
        for i in range(int(self.tv_ratio*self.sample_number)):
            self.train_image_list.append(self.raw_data[i][0])
            self.train_texts_list.append(self.raw_data[i][1])

        for i in range(int(self.tv_ratio*self.sample_number),self.sample_number):
            self.val_image_list.append(self.raw_data[i][0])
            self.val_texts_list.append(self.raw_data[i][1])
        
    def _batch_generator_train(self, batch_size):
        
        Image_Batch = np.zeros((batch_size, self.img_size, self.img_size, 3))
        Texts_Batch = np.zeros((batch_size, self.max_len))
        Margin_Batch = np.zeros((batch_size, 1))
        Y_Batch = np.zeros((batch_size, ))
        while True:
            for i in range(math.ceil( int(self.tv_ratio*self.sample_number) / batch_size)):
                sel = slice(i*batch_size, (i+1)*batch_size)
                image_path_batch = self.train_image_list[sel]
                texts_data_batch = self.train_texts_list[sel]

                for j in range(batch_size):
                    Image_Batch[j] = self._image_preprocessing(image_path_batch[j])
                    Texts_Batch[j] = self._text_preprocessing(texts_data_batch[j])
                    Margin_Batch[j] = self.margin
                
                yield [Image_Batch, Texts_Batch, Margin_Batch], Y_Batch
    
    def _batch_generator_val(self, batch_size):
        
        Image_Batch = np.zeros((batch_size, self.img_size, self.img_size, 3))
        Texts_Batch = np.zeros((batch_size, self.max_len))
        Margin_Batch = np.zeros((batch_size, 1))
        Y_Batch = np.zeros((batch_size, ))
        while True:
            for i in range(math.ceil( int((1-self.tv_ratio)*self.sample_number) / batch_size)):
                sel = slice(i*batch_size, (i+1)*batch_size)
                image_path_batch = self.val_image_list[sel]
                texts_data_batch = self.val_texts_list[sel]

                for j in range(batch_size):
                    Image_Batch[j] = self._image_preprocessing(image_path_batch[j])
                    Texts_Batch[j] = self._text_preprocessing(texts_data_batch[j])
                    Margin_Batch[j] = self.margin
                
                yield [Image_Batch, Texts_Batch, Margin_Batch], Y_Batch

        
if __name__ == "__main__":
    D = PairdataLoader('dataset/pair_train_data.p','dataset/vocabulary_list.p', 0.1)
    g = D._batch_generator(2)
    [a, b, c],y = next(g)
    print(a)
    print(b)
    print(c)

