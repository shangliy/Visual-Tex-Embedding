"""
File: configurations.py
Author: Shanglin Yang
Description: Configurations for model train and inference
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
      self.l1 = 0.01
      self.l2 = 0.01
      self._embedding_size = 256
      self._vocab_size = 821
      self.batch_size = 4
      self.epochs = 1

      self.img_size = 299
      self.max_len = 23
      self.tv_ratio = 0.9

      self.emb_margin = 0.1

      self.data_path = 'dataset/pair_train_data.p'
      self.vocab_path = 'dataset/vocabulary_list.p'