import tensorflow as tf
import numpy as np
import yaml
import os
import h5py
import time
import sys
import math

from Lib.Data_Processing import *
from Lib.Utility import *
from Model.MultiDim_Analyzer_Model import MultiDim_Analyzer
from Model.Base_Handler import Basement_Handler


class MDAnalyzer_Handler(Basement_Handler):
    def __init__(self, dataset_name, model_config, sess, is_training=True):
        
        # Initialization of Configuration, Parameter and Datasets
        super(MDAnalyzer_Handler, self).__init__(sess=sess, model_config=model_config, is_training=is_training)
        self.initial_parameter()
        self.data_assignment(dataset_name)
        
        # Define the general model and the corresponding input
        self.shape_enc = (self.batch_size, self.num_identity, self.period_enc)
        self.shape_dec = (self.batch_size, self.num_identity, self.period_dec)
        self.input_enc = tf.placeholder(tf.float32, shape=self.shape_enc, name='encoder_inputs')
        self.input_ori = tf.placeholder(tf.float32, shape=self.shape_dec, name='decoder_inputs')
        self.truth_pred = tf.placeholder(tf.float32, shape=self.shape_dec, name='ground_truth')
        self.truth_mask = tf.placeholder(tf.float32, shape=self.shape_dec, name='natural_missing')
        self.move_mask = tf.placeholder(tf.float32, shape=self.shape_dec, name='removed_missing')
        self.shared_info = tf.placeholder(tf.float32, shape=(self.num_identity, self.num_shared_feature), name='position')
        with tf.variable_scope("impute", reuse=tf.AUTO_REUSE):
            self.impute_segs = tf.get_variable("impute_var",shape=self.shape_enc, trainable=True,
                                               initializer=tf.random_normal_initializer(mean=0,stddev=0.1))
        # Initialization for the model training structure.
        self.learning_rate = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(self.lr_init),trainable=False)
        self.lr_new = tf.placeholder(tf.float32, shape=(), name='lr_new')
        self.lr_update = tf.assign(self.learning_rate, self.lr_new, name='lr_update')
        
        self.train_test_valid_assignment()
        self.trainable_parameter_info()
        self.saver = tf.train.Saver(tf.global_variables())

    def initial_parameter(self):
        
        # Configuration Set
        config = self.model_config
        
        # Model Input Initialization
        self.batch_size = int(config.get('batch_size',1))
        self.period_enc = int(config.get('period_enc',12))
        self.period_dec = int(config.get('period_dec',12))
        
        # Initialization for Training Controler
        self.epochs = int(config.get('epochs',100))
        self.epoch_iter = int(config.get('epoch_iter',5))
        self.patience = int(config.get('patience',30))
        self.lr_init = float(config.get('learning_rate',0.001))
        self.lr_decay = float(config.get('lr_decay',0.1))
        self.lr_decay_epoch = int(config.get('lr_decay_epoch',20))
        self.lr_decay_interval = int(config.get('lr_decay_interval',10))

    def data_assignment(self,dataset_name):
        model_config = self.model_config
        set_whole, self.node_pos, self.maximum = Data_Division(dataset_name)
        
        # Pre-calculation for model training
        self.scalar = limit_scalar(set_whole)
        self.set_whole,disp_whole = self.scalar.transform(set_whole)
        self.num_identity,self.num_shared_feature = self.node_pos.shape[0],self.node_pos.shape[1]
        self.whole_segs = (set_whole[0].shape[-1]-self.period_enc)/2+1
        self.whole_size = int(self.whole_segs/self.batch_size)
        
        # Display the data structure of Training/Testing/Validation Dataset
        print 'Available Segments[batches] %d[%d] Shape of data/mask piece %s and min-mean-max is %.2f-%.2f-%.2f' % (
            self.whole_segs,self.whole_size,set_whole[0].shape, disp_whole[0], disp_whole[1], disp_whole[2])
        print 'Measurement maximum(average,std) are %4.4f(%4.4f,%4.4f)' % (self.maximum,self.scalar.mean,self.scalar.std)
        
        # Data Generator
        self.gen_whole = Data_Generator(self.set_whole, set_whole[0], self.whole_segs, self.batch_size, self.period_enc, is_training=True)
        
    def train_test_valid_assignment(self):#, is_training = True, reuse = False
        
        # the original mask use 1 to indicate the missing point, and the inverse has been done during the data_division function
        value_sets = (
            tf.expand_dims(self.input_enc, -1),         # the input value of encoder (current data with randomly removed)
            tf.expand_dims(self.input_ori, -1),         # the input value of decoder (future/current data with randomly removed)
            tf.expand_dims(self.truth_pred,-1),         # the groundthuth of the future/current prediction
            tf.expand_dims(self.truth_mask,-1),         # the label of NATURAL missing data 0 -- missing, 1 -- available
            tf.expand_dims(self.move_mask, -1),         # the label of remove missing data 0 -- missing, 1 -- available
            self.shared_info,                     # the shared information of the nodes (position), normalized value [0,1]
            self.scalar                         # (Class Function) Rescale the the input and the output
        )
        with tf.name_scope('Train'):
            with tf.variable_scope('MultiDim_Analyzer', reuse=False):
                self.MD_Analyzer_train = MultiDim_Analyzer(value_sets, self.learning_rate, self.sess, self.model_config, is_training=True)
                
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        print ('Training Started')
        min_impute_metric = float('inf')
        epoch_cnt,wait = 0,0
        
        start_time = time.time()
        while epoch_cnt <= self.epochs:
            
            # Training Preparation: Learning rate pre=setting, Model Interface summary.
            
            cur_lr = self.calculate_scheduled_lr(epoch_cnt)
            whole_fetches = {'global_step': tf.train.get_or_create_global_step(),
                             'train_op': self.MD_Analyzer_train.train_op, 
                             'preds': self.MD_Analyzer_train.orig_impute,
                             'metric': self.MD_Analyzer_train.orig_metric,
                             'loss': self.MD_Analyzer_train.loss}
            Results = {"loss":[],"imputed":[],"metric":[],"ground":[],"mask_compare":[]}
            # Framework and Visualization SetUp for Training 
            for trained_batch in range(0,self.whole_size):
                (curdata,curmask,curmove,curdata_orig) = self.gen_whole.next()
                feed_dict_whole = {self.input_enc: curdata*curmove, 
                                   self.input_ori: curdata_orig, 
                                   self.truth_pred: curdata, 
                                   self.truth_mask: curmask,
                                   self.move_mask: curmove,
                                   self.shared_info: self.node_pos}
                whole_output = self.sess.run(whole_fetches,feed_dict=feed_dict_whole)
                message = "Epoch [%3d/%3d] [%d/%d] lr: %.4f, loss: %.8f" % (
                    epoch_cnt, self.epochs, trained_batch, self.whole_size, cur_lr, whole_output["loss"])
                if trained_batch % 50 == 0:
                    print message
                
                
                Results["metric"].append(whole_output['metric'])
                Results["loss"].append(whole_output['loss'])
                Results["imputed"].append(whole_output['preds'])
                Results["ground"].append(curdata_orig)
                Results["mask_compare"].append(curmask-curmove)
                global_step = whole_output['global_step']
            
            loss,metric_seg = np.mean(Results["loss"]),np.mean(Results["metric"],axis=0)
            if metric_seg[0] <= min_impute_metric:
                min_impute_metric = metric_seg[0]
            metrics = calculate_metrics_np(Results["imputed"],Results["ground"],Results["mask_compare"])
            
            # Information Logging for Model Training and Validation (Maybe for Curve Plotting)
            summary_format = ['loss/train_loss','metric/mse_segmin','metric/rmse','metric/mae','metric/mape','metric/mre']
            summary_data = [loss,min_impute_metric,metrics[1],metrics[2],metrics[3],metrics[4]]
            self.summary_logging(global_step, summary_format, summary_data)
            # Message Summary of each epoch (For info.log logging)
            message = 'Epoch [%3d/%3d] loss: %.4f(%.4f), Orig MSE/RMSE/MAE/MAPE/MRE %s' % (
                epoch_cnt, self.epochs, np.mean(Results["loss"]),min_impute_metric,metrics)
            self.logger.info(message)
            epoch_cnt += 1
        print '%ds'%(time.time()-start_time)
            
    def calculate_scheduled_lr(self, epoch, min_lr=1e-6):
        decay_factor = int(math.ceil((epoch - self.lr_decay_epoch) / float(self.lr_decay_interval)))
        new_lr = self.lr_init * self.lr_decay ** max(0, decay_factor)
        new_lr = max(min_lr, new_lr)
        
        self.logger.info('Current learning rate to: %.6f' % new_lr)
        sys.stdout.flush()
        
        self.sess.run(self.lr_update, feed_dict={self.lr_new: new_lr})
        self.MD_Analyzer_train.set_lr(self.learning_rate)
        return new_lr