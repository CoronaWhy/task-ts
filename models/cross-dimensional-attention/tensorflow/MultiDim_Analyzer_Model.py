import tensorflow as tf
import numpy as np
import math
import os
import json

from Lib.Utility import *
from Model.Base_TFModel import Basement_TFModel

class MultiDim_Analyzer(Basement_TFModel):
    
    def __init__(self, value_sets, init_learning_rate, sess, config, is_training=True, *args, **kwargs):
        
        super(MultiDim_Analyzer, self).__init__(sess=sess, config=config, learning_rate=init_learning_rate,is_training=is_training)
        '''
        Model input Explanation & Reminder:
        enc_input: the masked input of encoder, the dimension of time may varies with dec_input by [data generation and assignment in handler]
        dec_input: the masked input of encoder, the dimension of time may varies with enc_input by [data generation and assignment in handler]
        truth_pred: used in loss calculation: have identical dimension with decoder input, same(shifted) value for completion(prediction)
        truth_mask: used in the calculation of metric like MSE and relabel the model output for comparison (Mask of Natural Loss)
        shared_info: auxiliary information for encoder/decoder input encoding for identity dimension. BTW, encoding in time dimension is similar to NLP
        scalar: the maximum value of each measurement, used for metric calculation
        
        For data in the value_sets (Data Unzip):
        Size for model input/output: (batch_size, num_identity, num_measurement, period_enc/dec, 1)
        Size for auxiliary shared_info: (batch_size, num_shared_feature)
        '''
        (enc_input, ori_input, truth_pred, truth_mask, move_mask, shared_info, scalar) = value_sets
        self.num_identity = enc_input.get_shape().as_list()[1]

        # Initialization of the model hyperparameter, enc-dec structure, evaluation metric & Optimizier
        self.initial_parameter()
        self.model_input,self.model_output = self.encdec_handler(enc_input, shared_info, truth_pred, truth_mask, move_mask)
        self.metric_opt(self.model_output, ori_input, truth_pred, truth_mask, move_mask, scalar)

    def encdec_handler(self, enc_input, shared_info, truth_pred, truth_mask, move_mask):
        # for dimention introduction of the input: refer to class initialization
        
        # Options for Model Structure
        # Independent -------------- 3 Enc-Dec model are used to learn the relationship for each dimension repectively
        # Sequence ----------------- Following an arbitrary order to do the multiplication on the value vector
        # Element-wise-addition ---- Multiplication on the same value vector and then do the element-wise addition (average)
        # Concatenation ------------ Multiplication on the same value vector and then do the concatenation and matmul (generalized ew-addition)
        # Dimension-reduce --------- Expand the input 3D value to a 1D vector and then calculate the huge AM. (limitation of memory)
        
        # The encode is available for Identity&Measurement and Time dimension
        (shared_encoder,shared_decoder,time_encoder,time_decoder) = self.auxiliary_encode(shared_info) 
        if self.flag_casuality == True:
            mask_casuality = self.casual_mask()
        else:
            mask_casuality = None
        if self.flag_imputation == True:
            self.mask_imputation = self.impute_mask()
        else:
            self.mask_imputation = None
        if self.flag_time == True:
            enc_input = enc_input + time_encoder
        if self.flag_identity == True:
            enc_input = enc_input + shared_encoder
        
        with tf.variable_scope('layer_init'):
            enc_init = self.multihead_attention(enc_input, self.attention_unit, self.model_structure)
            enc_init = self.feed_forward_layer(enc_init, self.conv_unit, self.filter_encdec)
            enc_init = enc_input + tf.multiply(tf.constant(1.0,dtype=tf.float32)-move_mask,enc_init)
        topenc = self.encoder(enc_input, self.model_structure)
        return enc_init,topenc
        
    def encoder(self,enclayer_init,model_structure):
        enclayer_in = enclayer_init
        with tf.variable_scope('Encoder'):
            for cnt_enclayer in range(0,self.num_enclayer):
                with tf.variable_scope('layer_%d'%(cnt_enclayer)):
                    enclayer_in = self.layer_norm(enclayer_in + self.multihead_attention(
                        enclayer_in, self.attention_unit, model_structure), 'norm_1')
                    enclayer_in = self.layer_norm(enclayer_in + self.feed_forward_layer(
                        enclayer_in, self.conv_unit, self.filter_encdec), 'norm_2')#
        with tf.variable_scope('Enc_pred1'):
            enclayer_in = self.multihead_attention(enclayer_in, self.attention_unit, model_structure)
            enclayer_in = self.feed_forward_layer(enclayer_in, self.conv_unit, self.filter_encdec)
            return enclayer_in

    def decoder(self, declayer_init, model_structure, encoder_top, mask_casuality=None):
        declayer_in = declayer_init
        with tf.variable_scope('Decoder'):
            with tf.variable_scope('layer_0'):
                declayer_in = self.layer_norm(declayer_in + self.multihead_attention(
                    declayer_in, self.attention_unit, model_structure, mask=mask_casuality), 'norm_1')
                (attention_out,KVtop_share) = self.multihead_attention(
                    declayer_in, self.attention_unit, model_structure, top_encod=encoder_top, scope='enc-dec-attention')
                declayer_in = self.layer_norm(declayer_in + attention_out, 'norm_2')
                declayer_in = self.layer_norm(declayer_in + self.feed_forward_layer(
                    declayer_in, self.conv_unit, self.filter_encdec), 'norm_3')
            for cnt_declayer in range(1,self.num_declayer):
                with tf.variable_scope('layer_%d'%(cnt_declayer)):
                    declayer_in = self.layer_norm(declayer_in + self.multihead_attention(
                        declayer_in, self.attention_unit, model_structure, mask=mask_casuality), 'norm_1')
                    declayer_in = self.layer_norm(declayer_in + self.multihead_attention(
                        declayer_in, self.attention_unit, model_structure, top_encod=encoder_top, cache=KVtop_share, 
                        scope='enc-dec-attention'), 'norm_2')
                    declayer_in = self.layer_norm(declayer_in + self.feed_forward_layer(
                        declayer_in, self.conv_unit, self.filter_encdec), 'norm_3')
        with tf.variable_scope('Dec_pred_1'):
            declayer_in = self.multihead_attention(declayer_in, self.attention_unit, model_structure)
            declayer_in = self.feed_forward_layer(declayer_in, self.conv_unit, self.filter_encdec)
        
            return declayer_in

    def metric_opt(self, model_output, truth_orig, truth_pred, truth_mask, move_mask, scalar):
        
        loss_mask = move_mask
        global_step = tf.train.get_or_create_global_step()
        avail_output = tf.multiply(model_output, loss_mask)
        avail_truth = tf.multiply(truth_pred, loss_mask)
        
        if self.loss_func == 'MSE':
            self.loss = loss_mse(avail_output, avail_truth, loss_mask)
        elif self.loss_func == 'RMSE':
            self.loss = loss_rmse(avail_output, avail_truth, loss_mask)
        elif self.loss_func == 'MAE':
            self.loss = loss_mae(avail_output, avail_truth, loss_mask)
        else:
            self.loss = loss_rmse(avail_output, avail_truth, loss_mask)+loss_mae(avail_output, avail_truth, loss_mask)
            
        if self.is_training:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
        self.info_merge = tf.summary.merge_all()
        
        orig_preds = scalar.TFinverse_transform(model_output)
        orig_truth = tf.multiply(truth_orig,truth_mask-move_mask)
        self.orig_impute = tf.multiply(truth_orig,move_mask) + tf.multiply(orig_preds, tf.constant(1.0,dtype=tf.float32)-move_mask)
        self.orig_metric = calculate_metrics(tf.multiply(orig_preds,truth_mask-move_mask), orig_truth, truth_mask-move_mask)
        
    def multihead_attention(self, att_input, att_unit, model_structure, top_encod=None, cache=None, mask=None, scope='self-attention'):
        """
        att_input: the input to be calculated in this module with size of [batch, #identity, #measurement, time, 1]
        att_unit:  the hyperparameter for the dimention of Q/K and V
        top_encod: the output from the top encoder layer [batch_size,#identity, #measurement, time, 1]
        mask: mask the casuality of the self-attention layer, [batch, time, time] or [batch, #id*#meas*time, #id*#meas*time]
        
        3D convolution is applied to realize the unit expansion. For the convenience of application, we have the following index mapping:
        [batch, in_depth, in_height, in_width, in_channels] = [batch_size, num_identity, num_measurement, length_time, 1]
        """
        # Initialization for some necessary item
        (value_units, Iunits, Tunits) = att_unit
        KVtop_cache = None
        # Since the value of num_measurement and period are small and may equal to each other by coincidence
        # We use the dimention of num_identity
        V_filters, V_kernal, V_stride = value_units*self.num_heads, (1, self.V_timfuse), (1, self.V_timjump)
        batch,ids,time = att_input.get_shape().as_list()[:3]

        with tf.variable_scope(scope):
            if top_encod is None or cache is None:
                if top_encod is None:
                    top_encod = att_input
                else:
                    if cache is None:
                        KVtop_cache = {}
                # Linear projection (unit expansion) for multihead-attention dimension: 
                Q_iden = tf.layers.dense(tf.reshape(att_input,[batch,ids,-1]), self.num_heads*Iunits, 
                                         use_bias=False, name='Q_ID')
                K_iden = tf.layers.dense(tf.reshape(top_encod,[batch,ids,-1]), self.num_heads*Iunits, 
                                         use_bias=False, name='K_ID')
                Q_time = tf.layers.dense(tf.reshape(tf.transpose(att_input,[0,2,1,3]),[batch,time,-1]), self.num_heads*Tunits, 
                                         use_bias=False, name='Q_Time')
                K_time = tf.layers.dense(tf.reshape(tf.transpose(top_encod,[0,2,1,3]),[batch,time,-1]), self.num_heads*Tunits, 
                                         use_bias=False, name='K_Time')
                V = tf.layers.conv2d(inputs=top_encod, filters=V_filters, kernel_size=V_kernal, strides=V_stride, 
                                     padding="same", data_format="channels_last", name='V')
                if KVtop_cache is not None:
                    KVtop_cache = {'share_Kid':K_id, 'share_Ktime':K_time, 'share_V':V}
            else:
                Q_iden = tf.layers.dense(tf.reshape(att_input,[batch,ids,-1]), self.num_heads*Iunits, 
                                         use_bias=False, name='Q_ID')
                Q_time = tf.layers.dense(tf.reshape(tf.transpose(att_input,[0,2,1,3]),[batch,time,-1]), self.num_heads*Tunits, 
                                         use_bias=False, name='Q_Time')
                K_id,K_time,V = cache['share_Kid'],cache['share_Ktime'],cache['share_V']

            # Split the matrix to multiple heads and then concatenate to build a larger batch size: 
            # [self.batch_size*self.num_heads, self.X, self.X]
            Qhb_id = tf.concat(tf.split(Q_iden, self.num_heads, axis=2), axis=0)
            Qhb_time = tf.concat(tf.split(Q_time, self.num_heads, axis=2), axis=0)
            Khb_id = tf.concat(tf.split(K_iden, self.num_heads, axis=2), axis=0)
            Khb_time = tf.concat(tf.split(K_time, self.num_heads, axis=2), axis=0)
            # [self.batch_size*self.num_heads, self.num_identity, self.num_measurement, self.length_time, 'hidden-units']
            Q_headbatch = (Qhb_id,Qhb_time)
            K_headbatch = (Khb_id,Khb_time)
            V_headbatch = tf.concat(tf.split(V, self.num_heads, axis=3), axis=0)
            
            if mask is not None:
                mask_recur = tf.tile(mask, [self.num_heads, 1, 1])
            else:
                mask_recur = None
            
            out = self.softmax_combination(Q_headbatch, K_headbatch, V_headbatch, model_structure, att_unit, mask_recur)

            # Merge the multi-head back to the original shape 
            # [batch_size, self.num_identity, self.num_measurement, self.length_time, 'hidden-units'*self.num_heads]
            out = tf.concat(tf.split(out, self.num_heads, axis=0), axis=3)  # 
            out = tf.layers.dense(out, 1, name='multihead_fuse')
            out = tf.layers.dropout(out, rate=self.attdrop_rate, training=self.is_training)
            
            if KVtop_cache is None:
                return out
            else:
                return (out,KVtop_cache)
    
    def feed_forward_layer(self, info_attention, num_hunits, filter_type='dense'):
        '''
        forward_type: 
        "dense" indicates dense layer, 
        "graph" indicates graph based FIR filter (graph convolution),
        "attention" indicates applying the attention algorithm
        "conv" indicates the shared convolution kernal is applied instead of a big weight matrix
        self.ffndrop_rate may be considered later 03122019
        '''
        channel = info_attention.get_shape().as_list()[-1]
        if filter_type == 'dense':
            ffn_dense = tf.layers.dense(info_attention, num_hunits, use_bias=True, activation=tf.nn.relu, name=filter_type+'1')
            ffn_dense = tf.layers.dense(info_attention, num_hunits, use_bias=True, activation=None, name=filter_type+'2')
            return tf.layers.dense(ffn_dense, channel, use_bias=True, activation=None, name=filter_type+'3')
        elif filter_type == 'graph': 
            raise NotImplementedError
        elif filter_type == 'attention':
            raise NotImplementedError
        elif filter_type == 'conv':
            raise NotImplementedError
    
    def layer_norm(self, norm_input, name_stage):
        norm_step = tf.contrib.layers.layer_norm(tf.transpose(tf.squeeze(norm_input),perm=[0,2,1]), 
                                                 begin_norm_axis=2, center=True, scale=True,scope=name_stage)
        return tf.expand_dims(tf.transpose(norm_step,perm=[0,2,1]),-1)
    
    def softmax_combination(self, Q, K, V, model_structure, att_unit, mask=None):
        '''mask is applied before the softmax layer, no dropout is applied, '''
        value_units,segs = V.get_shape().as_list()[-1],V.get_shape().as_list()[0]
        ids,time = Q[0].get_shape().as_list()[1],Q[1].get_shape().as_list()[1]
        (value_units, Iunits, Tunits) = att_unit
        
        (Q_I,Q_T) = Q
        (K_I,K_T) = K

        # Check the dimension consistency of the combined matrix
        assert Q_I.get_shape().as_list()[1:] == K_I.get_shape().as_list()[1:]
        assert Q_T.get_shape().as_list()[1:] == K_T.get_shape().as_list()[1:]
        assert Q_I.get_shape().as_list()[0] == Q_T.get_shape().as_list()[0]
        assert K_I.get_shape().as_list()[0] == K_T.get_shape().as_list()[0]

        # Build the Attention Map
        AM_Identity = tf.matmul(Q_I, tf.transpose(K_I, [0, 2, 1])) / tf.sqrt(tf.cast(Iunits, tf.float32))
        AM_Time = tf.matmul(Q_T, tf.transpose(K_T, [0, 2, 1])) / tf.sqrt(tf.cast(Tunits, tf.float32))
        if mask is not None:
            AM_Time = tf.multiply(AM_Time,mask) + tf.constant(-np.inf)*(tf.constant(1.0)-mask)
        if self.mask_imputation is not None:
            (iden_mask, time_mask) = self.mask_imputation
            #AM_Identity = tf.multiply(AM_Identity,iden_mask) + tf.constant(-1.0e9)*(tf.constant(1.0)-iden_mask)
            AM_Time = tf.multiply(AM_Time,time_mask) + tf.constant(-1.0e9)*(tf.constant(1.0)-time_mask)
        AM_Identity = tf.nn.softmax(AM_Identity, 2)
        AM_Time = tf.nn.softmax(AM_Time, 2)

        shape_id = [segs, ids, time, value_units]
        shape_time = [segs, time, ids, value_units]

        if model_structure == 'Sequence':
            Out_Id = tf.reshape(tf.matmul(AM_Identity, tf.reshape(V,[segs, ids, -1])), shape_id)
            Out_Id = tf.transpose(Out_Id,perm=[0,2,1,3])
            Out_Id_Time = tf.reshape(tf.matmul(AM_Time, tf.reshape(Out_Id, [segs,time,-1])), shape_time)
            return tf.transpose(Out_Id_Time,perm=[0,2,1,3])
        else:
            V_id,V_time = V,tf.transpose(V,perm=[0,2,1,3])
            Out_Identity = tf.reshape(tf.matmul(AM_Identity, tf.reshape(V_id,[segs, ids, -1])), shape_id)
            Out_Time = tf.reshape(tf.matmul(AM_Time, tf.reshape(V_time,[segs, time, -1])), shape_time)

            Out_Time = tf.transpose(Out_Time,perm=[0,2,1,3])
            if model_structure == 'Element-wise-addition':
                return tf.divide(tf.add(Out_Identity,Out_Time),tf.constant(2.0))
            elif model_structure == 'Concatenation':
                Attention_output = tf.concat([Out_Identity, Out_Time], 3)
                return tf.layers.dense(Attention_output, value_units, use_bias=False)
            else:
                raise UnavailableStructureMode
                    
    def casual_mask(self):
        '''
        This function is only applied in the self-attention layer of decoder.
        The lower triangular matrix is used to indicate the available reference of all position in each calculation
        Key Idea: Only the previous position is applied to predict the future
        '''
        batch_size,period = self.batch_size,self.period_dec
        casual_unit = np.tril(np.ones((period, period)))
        casual_tensor = tf.convert_to_tensor(casual_unit, dtype=tf.float32)
        return tf.tile(tf.expand_dims(casual_tensor, 0), [batch_size, 1, 1])
    
    def impute_mask(self):
        batch_size = self.batch_size
        iden_unit = 1.0-np.identity(self.num_identity)
        time_unit = 1.0-np.identity(self.period_dec)
        iden_tensor = tf.tile(tf.expand_dims(tf.convert_to_tensor(iden_unit, dtype=tf.float32), 0), [self.num_heads*batch_size, 1, 1])
        time_tensor = tf.tile(tf.expand_dims(tf.convert_to_tensor(time_unit, dtype=tf.float32), 0), [self.num_heads*batch_size, 1, 1])
        return (iden_tensor,time_tensor)

    def initial_parameter(self):

        config = self.config
        # Parameter Initialization of Data Assignment
        self.batch_size = int(config.get('batch_size',1))
        self.period_enc = int(config.get('period_enc',12))
        self.period_dec = int(config.get('period_dec',12))
        
        # Parameter Initialization of Model Framework
        self.num_heads = int(config.get('num_heads',8))
        self.num_enclayer = int(config.get('num_enclayer',5))
        self.num_declayer = int(config.get('num_declayer',5))
        self.model_structure = self.config.get('model_structure')
        
        # Parameter Initialization of Attention (Q K V)
        self.AM_timjump = int(config.get('time_stride_AM',1))
        self.V_timjump = int(config.get('time_stride_V',1))
        self.AM_timfuse = int(config.get('time_fuse_AM',1))
        self.V_timfuse = int(config.get('time_fuse_V',1))
        vunits,Iunits,Tunits = int(config.get('units_value',6)),int(config.get('units_IDw',14)),int(config.get('units_Timew',6))
        self.attention_unit = (vunits, Iunits, Tunits)
        
        # Parameter Initialization of Filter (Enc-Dec, Prediction)
        self.filter_encdec = config.get('filter_encdec','dense')
        self.conv_unit = int(config.get('units_conv',4))
        self.attdrop_rate = float(config.get('drop_rate_attention',0.0))
        self.ffndrop_rate = float(config.get('drop_rate_forward',0.1))
        self.filter_pred = config.get('filter_pred','dense')
        self.pred_unit = int(config.get('units_pred',8))
        
        # label of mask
        self.flag_identity = config.get('flag_identity',False)
        self.flag_time = config.get('flag_time',False)
        self.flag_casuality = config.get('flag_casuality',False)
        self.flag_imputation = config.get('flag_imputation',False)

    def auxiliary_encode(self,shared_info):
        # The concatenation is not applicable in this part since all the attention of all three dimension need to be learned.
        # Expanding each dimention will not make sense for our model.
        # Concatenation with the feature dimention (expanded as 1) is equivalent with the element-wise addition.
        with tf.variable_scope('shared_feature'):
            shared_encoder = tf.layers.dense(tf.expand_dims(shared_info,0), self.period_enc, 
                                             use_bias=False, activation=None, name='encoder')
            shared_encoder = tf.reshape(shared_encoder, [1, self.num_identity, self.period_enc, 1])
            shared_encoder = tf.tile(shared_encoder, [self.batch_size, 1, 1, 1])
            shared_decoder = tf.layers.dense(tf.expand_dims(shared_info,0), self.period_dec, 
                                             use_bias=False, activation=None, name='decoder')
            shared_decoder = tf.reshape(shared_decoder, [1, self.num_identity, self.period_dec, 1])
            shared_decoder = tf.tile(shared_decoder, [self.batch_size, 1, 1, 1])
            
            denom = tf.constant(1000.0)
            phase_enc = tf.linspace(0.0,self.period_enc-1.0,self.period_enc)*tf.constant(math.pi/180.0)/denom
            phase_dec = tf.linspace(0.0,self.period_dec-1.0,self.period_dec)*tf.constant(math.pi/180.0)/denom
            sin_enc,sin_dec = tf.expand_dims(tf.sin(phase_enc),0),tf.expand_dims(tf.sin(phase_dec),0)
            
            time_encoder = tf.expand_dims(tf.tile(tf.expand_dims(sin_enc,0),[self.batch_size,self.num_identity,1]),-1)
            time_decoder = tf.expand_dims(tf.tile(tf.expand_dims(sin_dec,0),[self.batch_size,self.num_identity,1]),-1)
            return (shared_encoder,shared_decoder,time_encoder,time_decoder)