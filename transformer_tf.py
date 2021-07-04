# -*- coding: utf-8 -*-
'''
制作:路过的小林
参考代码:莫烦PYTHON 苏剑林bert4keras
'''
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
class Layer(keras.layers.Layer):
         def __init__(self, **kwargs):
             super(Layer, self).__init__(**kwargs)
             self.supports_masking = True    
keras.layers.Layer=   Layer              
class LayerNormalization(keras.layers.Layer):
    """(Conditional) Layer Normalization
    代码来自苏剑林bert4keras
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = keras.activations.get(hidden_activation)
        self.hidden_initializer = keras.initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = keras.layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

   # @recompute_grad
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': keras.activations.serialize(self.hidden_activation),
            'hidden_initializer':
                keras.initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class MultiHead(keras.layers.Layer):
    #标准多头注意力层
    def __init__(self, n_head, 
                 head_dim,
                 drop_rate, 
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 use_rotary=False,
                 **kwargs):
        '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
         use_rotary 是否使用旋转位置编码
        '''
        super(MultiHead, self).__init__(**kwargs)
        self.head_dim = head_dim
        self.n_head = n_head
        self.drop_rate=drop_rate       
        self.use_bias=use_bias
        self.kernel_initializer= keras.initializers.get(kernel_initializer)
        self.use_rotary=use_rotary
    def build(self,input_shape):
        self.wq = keras.layers.Dense(self.n_head * self.head_dim,use_bias=self.use_bias,kernel_initializer=self. kernel_initializer)
        self.wk = keras.layers.Dense(self.n_head * self.head_dim,use_bias=self.use_bias,kernel_initializer=self. kernel_initializer)
        self.wv = keras.layers.Dense(self.n_head * self.head_dim,use_bias=self.use_bias,kernel_initializer=self. kernel_initializer)      # [n, step, h*h_dim]
        self.o_dense = keras.layers.Dense(self.n_head * self.head_dim,use_bias=self.use_bias,kernel_initializer=self. kernel_initializer)
        self.o_drop = keras.layers.Dropout(rate=self.drop_rate)
        super(MultiHead, self).build(input_shape)
    def call(self, x, mask):
        q,k,v=x[0],x[1],x[2]
        _q = self.wq(q)      # [n, q_step, h*h_dim]
        _k, _v = self.wk(k), self.wv(v)     # [n, step, h*h_dim]
        
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(_v)  # [n, h, step, h_dim]
        if self.use_rotary:
            _q,_k=self.roform_position(_q),self.roform_position(_k)
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)     # [n, q_step, h*dv]
        o = self.o_dense(context)       # [n, step, dim]
        o = self.o_drop(o)
        return o

    def split_heads(self, x):
        x =K.reshape(x, (K.shape(x)[0], K.shape(x)[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])       # [n, h, step, h_dim]

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        #可以通过改写这层实现自定义注意力层
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)  # [n, h_dim, q_step, step]
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)                               # [n, h, q_step, step]
        context = tf.matmul(self.attention, v)         # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        context = tf.transpose(context, perm=[0, 2, 1, 3])   # [n, q_step, h, dv]
        context = K.reshape(context, (K.shape(context)[0], K.shape(context)[1],self.n_head * self.head_dim))     # [n, q_step, h*dv]
        return context
    def roform_position(self,x):
        batch_size,n_head,sequence_length,diemension=K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],K.shape(x)[3]
        d=K.cast(diemension,dtype=K.floatx())
        position_ids = K.arange(0, sequence_length, dtype=K.floatx())[None]
        indices = K.arange(0, diemension, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / d)
        indices = tf.einsum('bn,d->bnd', position_ids, indices)
        sin=K.sin(indices)
        cos=K.cos(indices)
        x_=K.stack([-1*x[...,1::2],x[...,::2]],4)
        x_=K.reshape(x_,[batch_size,n_head,sequence_length,diemension])
        return cos*x+sin*x_
    def get_config(self):
        config = {
            'n_head': self.n_head,
            'head_dim': self.head_dim,
            'drop_rate': self.drop_rate,
            'use_bias': self.use_bias,
            'kernel_initializer':keras.initializers.serialize(self.kernel_initializer),
            'use_rotary':self.use_rotary
        }
        base_config = super(MultiHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim,
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 **kwargs):
        '''
         model_dim:全连接层的维度
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
        super(PositionWiseFFN,self).__init__(**kwargs)
        self.use_bias=use_bias
        self.kernel_initializer=  keras.initializers.get(kernel_initializer)
        self.model_dim=model_dim
    def build(self, input_shape):
        super(PositionWiseFFN, self).build(input_shape)  
        self.l = keras.layers.Dense(self.model_dim*4, activation=keras.activations.relu,use_bias=self.use_bias,kernel_initializer=self.kernel_initializer)
        self.o = keras.layers.Dense(self.model_dim,use_bias=self.use_bias,kernel_initializer=self. kernel_initializer)
    def call(self, x):
        o = self.l(x)
        o = self.o(o)
        return o         # [n, step, dim]
    def get_config(self):
        config = {
            'model_dim': self.model_dim,
            'use_bias': self.use_bias,
            'kernel_initializer':keras.initializers.serialize(self.kernel_initializer)
        }
        base_config = super(PositionWiseFFN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class EncodeLayer(keras.layers.Layer):
    def __init__(self,  n_head, 
                 head_dim,
                 drop_rate, 
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 use_rotary=False,
                 **kwargs):
        '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
        super(EncodeLayer,self).__init__(**kwargs)
        self.head_dim = head_dim
        self.n_head = n_head
        self.drop_rate=drop_rate       
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.use_rotary=use_rotary
    def build(self, input_shape):
        self.ln = [LayerNormalization() for _ in range(2)]  # only norm z-dim
        self.mh = MultiHead(self.n_head, 
                                 self.head_dim,
                                 self.drop_rate, 
                                 self.kernel_initializer,
                                 self.use_bias,
                                 self.use_rotary)
        self.ffn = PositionWiseFFN(self.n_head*self.head_dim,
                                   self.kernel_initializer,
                                   self.use_bias)
        self.drop = keras.layers.Dropout(self.drop_rate)
        super(EncodeLayer, self).build(input_shape)  
    def call(self, xz,  mask):
        attn = self.mh([xz, xz, xz], mask)       # [n, step, dim]
        o1 = self.ln[0](keras.layers.Add()([attn,xz]))
        ffn = self.drop(self.ffn(o1))
        o = self.ln[1](keras.layers.Add()([ffn,o1]))         # [n, step, dim]
        return o
    def get_config(self):
        config = {
            'n_head': self.n_head,
            'head_dim': self.head_dim,
            'drop_rate': self.drop_rate,
            'use_bias': self.use_bias,
            'kernel_initializer':self.kernel_initializer,
            'use_rotary':self.use_rotary
        }
        base_config = super(EncodeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Encoder(keras.layers.Layer):
    def __init__(self,  n_head, 
                 head_dim,
                 drop_rate, 
                 n_layer=1,
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 use_rotary=False,
                 **kwargs):
        '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
         n_layer：一个encoder所需要堆叠的encoder_layer数
        '''
        super(Encoder,self).__init__(**kwargs)
        self.head_dim = head_dim
        self.n_head = n_head
        self.drop_rate=drop_rate       
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer  
        self.n_layer=n_layer
        self.use_rotary=use_rotary
    def build(self, input_shape):
        self.ls = [EncodeLayer(self.n_head, 
                 self.head_dim,
                 self.drop_rate, 
                 self.kernel_initializer,
                 self.use_bias,
                 self.use_rotary) for _ in range(self.n_layer)]
        super(Encoder, self).build(input_shape)  
    def call(self, xz,mask):
        for l in self.ls:
            xz = l(xz,mask)
        return xz       # [n, step, dim]
    def get_config(self):
        config = {
            'n_head': self.n_head,
            'head_dim': self.head_dim,
            'drop_rate': self.drop_rate,
            'use_bias': self.use_bias,
            'kernel_initializer':self.kernel_initializer,
            'n_layer':self.n_layer,
            'use_rotary':self.use_rotary,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class DecoderLayer(keras.layers.Layer):
    '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
    def __init__(self,  n_head, 
                 head_dim,
                 drop_rate, 
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 use_rotary=False,
                 **kwargs):
        '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
        super(DecoderLayer,self).__init__(**kwargs)
        self.head_dim = head_dim
        self.n_head = n_head
        self.drop_rate=drop_rate       
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.use_rotary=use_rotary  
    def build(self, input_shape):
        self.ln = [LayerNormalization() for _ in range(3)] # only norm z-dim
        self.drop = keras.layers.Dropout(self.drop_rate)
        self.mh = [MultiHead(self.n_head, 
                                 self.head_dim,
                                 self.drop_rate, 
                                 self.kernel_initializer,
                                 self.use_bias,
                                 self.use_rotary) for _ in range(2)]
        self.ffn = PositionWiseFFN(self.n_head*self.head_dim,
                                   self.kernel_initializer,
                                   self.use_bias)
        super(DecoderLayer, self).build(input_shape)  
    def call(self, inputs, yz_look_ahead_mask, xz_pad_mask):
        yz, xz=inputs[0],inputs[1]
        attn = self.mh[0]([yz, yz, yz], yz_look_ahead_mask)       # decoder self attention
        o1 = self.ln[0](attn+yz )
        attn = self.mh[1]([o1, xz, xz], xz_pad_mask)       # decoder + encoder attention
        o2 = self.ln[1](keras.layers.Add()([attn,o1]) )
        ffn = self.drop(self.ffn(o2))
        o = self.ln[2](keras.layers.Add()([ffn ,o2]))
        return o
    def get_config(self):
        config = {
            'n_head': self.n_head,
            'head_dim': self.head_dim,
            'drop_rate': self.drop_rate,
            'use_bias': self.use_bias,
            'kernel_initializer':self.kernel_initializer,
            "use_rotary":self.use_rotary
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Decoder(keras.layers.Layer):
    '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
         n_layer：一个encoder所需要堆叠的encoder_layer数
        '''
    def __init__(self, n_head, 
                 head_dim,
                 drop_rate, 
                 n_layer=1,
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 use_rotary=False,
                 **kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.head_dim = head_dim
        self.n_head = n_head
        self.drop_rate=drop_rate       
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer  
        self.n_layer=n_layer
        self.use_rotary=use_rotary
    def build(self, input_shape):
        self.ls = [DecoderLayer(self.n_head, 
                 self.head_dim,
                 self.drop_rate, 
                 self.kernel_initializer,
                 self.use_bias,
                 self.use_rotary) for _ in range(self.n_layer)]
        super(Decoder, self).build(input_shape)      
    def call(self, inputs, yz_look_ahead_mask, xz_pad_mask):
        #yz代表目标，xz代表encoder的结果
        yz, xz=inputs[0],inputs[1]
        for l in self.ls:
            yz = l([yz, xz], yz_look_ahead_mask, xz_pad_mask)
        return yz
    def get_config(self):
        config = {
            'n_head': self.n_head,
            'head_dim': self.head_dim,
            'drop_rate': self.drop_rate,
            'use_bias': self.use_bias,
            'kernel_initializer':self.kernel_initializer,
            'n_layer':self.n_layer,
            'use_rotary':self.use_rotary,
        }
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class PositionEmbedding(keras.layers.Layer):
    """定义可训练的位置Embedding
    代码来自bert4keras
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        hierarchical=None,
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        '''
        input_dim：输入维度
        output_dim：输出维度
        merge_mode：连接的方法，默认是add，还可以"zero"返回embeding,"mul"使用乘性,
        hierarchical：是否使用层次位置
        embeddings_initializer：初始化方法,
        custom_position_ids：是否使用自定义的位置,
        '''
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, 'int32')
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = K.gather(embeddings, position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = K.gather(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            raise("input must is add or mul or zero")

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                keras.initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SinusoidalPositionEmbedding(keras.layers.Layer):
    """定义Sin-Cos位置Embedding
    代码来自bert4keras
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False, **kwargs
    ):
        '''
        output_dim：输出维度
        merge_mode：连接的方法，默认是add，还可以"zero"返回embeding,"mul"使用乘性,
        custom_position_ids：是否使用自定义的位置,
        '''
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            raise("input must is add or mul or zero")

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))