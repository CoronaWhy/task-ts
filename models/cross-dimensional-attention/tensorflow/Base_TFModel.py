# uncompyle6 version 3.7.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
# [GCC 8.4.0]
# Embedded file name: /home/jiawei/Tensor_MultiDim_NYC/Model/Base_TFModel.py
# Compiled at: 2019-04-29 02:51:25
import os, tensorflow as tf

class Basement_TFModel(object):
    """Define and Initialize the basic/necessary element of a tensorflow model """
    __module__ = __name__

    def __init__(self, sess, config, learning_rate, is_training):
        self.sess = sess
        self.config = config
        self.is_training = is_training
        self.model_name = config.get('model_name', 'MDAnalyzer')
        self.train_op = None
        self.learning_rate = learning_rate
        self.max_grad_norm = float(config.get('max_grad_norm', 5.0))
        self.loss = None
        self.loss_func = config.get('loss_func', 'RMSE')
        self.maximum_type = int(config.get('upbound', 1))
        return

    def set_lr(self, new_learning_rate):
        self.learning_rate = new_learning_rate

    def save_checkpoint(self, step=None):
        pass

    def load_checkpoint(self, step=None):
        pass

    def initial_parameter(self):
        pass