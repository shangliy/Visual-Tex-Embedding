from keras import backend as K
import tensorflow as tf

from keras.layers import Input
from keras.engine.topology import Layer
import numpy as np

def contrast_loss(y_true, y_pred):
    return K.mean(y_pred, axis=-1)

class ContrastiveLoss(Layer):
    
    def __init__(self,  **kwargs):
        
        super(ContrastiveLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ContrastiveLoss, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        words_embedding, image_embedding, emb_margin = x

        scores = tf.matmul(words_embedding, image_embedding,
                         transpose_a=False, transpose_b=True, name="scores")
    
        diagonal = tf.expand_dims(tf.diag_part(scores), 1)
        
        cost_s = tf.maximum(0.0, emb_margin - diagonal + scores)
        
        cost_im = tf.maximum(0.0,
                    emb_margin - tf.transpose(diagonal) + scores)
        cost_s = cost_s - tf.diag(tf.diag_part(cost_s))
        cost_im = cost_im - tf.diag(tf.diag_part(cost_im))
      
        emb_batch_loss = tf.reduce_sum(cost_s,axis=0) + tf.reduce_sum(cost_im,axis=0)
        
        return emb_batch_loss

    def compute_output_shape(self, input_shape):
        return input_shape[2]

if __name__ == '__main__':
    img_ = Input(shape=[256])
    txt_ = Input(shape=[256])
    em = Input(shape=[1])
    score = ContrastiveLoss()([img_,txt_,em])
    print(score)