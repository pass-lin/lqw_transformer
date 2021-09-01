# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:42:05 2021
self.args
@author: Administrator
"""

from lqw_transformer.layers import *
class EncodeLayer(keras.layers.Layer):
    def __init__(self,  n_head=None, 
                 head_dim=None,
                 drop_rate=None, 
                 kernel_initializer='glorot_uniform',
                 use_bias=True,
                 use_rotary=False,
                 use_Time_shift=False,
                 mask_future=False,
                 activation=keras.activations.relu,
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
        self.use_Time_shift=use_Time_shift
        self.attention= MultiHead
        self.mask=mask_future
        self.activation=keras.layers.Activation(activation)
    def time_shift_pad(self,x):
        return K.temporal_padding(x,(1,0))
    def time_shift(self,x):
        d=self.head_dim*self.n_head
        x=K.concatenate([self.time_shift_pad(x)[:,:-1,:d//2],x[:,:,d//2:]],-1 )
        return x
    def get_attention(self):
        layer= self.attention(n_head=self.n_head, 
                                head_dim=self.head_dim,
                                drop_rate=self.drop_rate, 
                                kernel_initializer=self.kernel_initializer,
                                use_bias=self.use_bias,
                                use_rotary=self.use_rotary,
                                use_Time_shift=self.use_Time_shift,
                                mask_future=self.mask,
                                )
        return layer
    def set_weight(self,input_shape):
        self.ln = [LayerNormalization() for _ in range(2)]  # only norm z-dim
        self.mh = self.get_attention()
        self.ffn = PositionWiseFFN(self.n_head*self.head_dim,
                                   self.kernel_initializer,
                                   self.use_bias,
                                   activation=self.activation)
        self.drop = keras.layers.Dropout(self.drop_rate)
    def build(self, input_shape):
        self.set_weight(input_shape)
        super(EncodeLayer, self).build(input_shape)  
    def call(self, xz,  mask):
        if self.use_Time_shift:
            xz=self.time_shift(xz)
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
            'use_rotary':self.use_rotary,
            'use_Time_shift':self.use_Time_shift,
            'mask_future':self.mask,
            'activation':keras.activations.serialize(self.activation),
        }
        base_config = super(EncodeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Encoder(EncodeLayer):
    def __init__(self, 
                 n_layer=1,
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
        self.attention=EncodeLayer
        self.n_layer=n_layer
    def set_weight(self,input_shape):
        self.ls = [ self.get_attention() for _ in range(self.n_layer)]
    def call(self, xz,mask=None):
        for l in self.ls:
            xz = l(xz,mask)
        return xz       # [n, step, dim]
    def get_config(self):
        config = {
            'n_layer':self.n_layer,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class DecoderLayer(EncodeLayer):
    '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
    def __init__(self,**kwargs):
        super(DecoderLayer,self).__init__(**kwargs)
        self.args=kwargs
    def set_weight(self,input_shape):
        self.ln = [LayerNormalization() for _ in range(3)] # only norm z-dim
        self.drop = keras.layers.Dropout(self.drop_rate)
        att2=self.get_attention()
        self.mask=True
        att1=self.get_attention()
        self.mh = [att1,att2]
        self.ffn = PositionWiseFFN(self.n_head*self.head_dim,
                                   self.kernel_initializer,
                                   self.use_bias,activation=self.activation)
    def call(self, inputs, yz_look_ahead_mask, xz_pad_mask):
        yz, xz=inputs[0],inputs[1]
        if self.use_Time_shift:
            xz=self.time_shift(xz)
            yz=self.time_shift(yz)
        attn = self.mh[0]([yz, yz, yz], yz_look_ahead_mask)       # decoder self attention
        o1 = self.ln[0](attn+yz )
        attn = self.mh[1]([o1, xz, xz], xz_pad_mask)       # decoder + encoder attention
        o2 = self.ln[1](keras.layers.Add()([attn,o1]) )
        ffn = self.drop(self.ffn(o2))
        o = self.ln[2](keras.layers.Add()([ffn ,o2]))
        return o
class Decoder(Encoder):
    '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
         n_layer：一个encoder所需要堆叠的encoder_layer数
        '''
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.attention=DecoderLayer
    def call(self, inputs, yz_look_ahead_mask, xz_pad_mask):
        #yz代表目标，xz代表encoder的结果
        yz, xz=inputs[0],inputs[1]
        for l in self.ls:
            yz = l([yz, xz], yz_look_ahead_mask, xz_pad_mask)
        return yz
class Free_EncoderLayer(EncodeLayer):
    def __init__(self, out_dims,
                     max_len,
                     **kwargs):
        '''
         max_len:序列的最大长度
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
        super(Free_EncoderLayer, self).__init__(**kwargs)
        self.out_dims = out_dims
        self.max_len=max_len
        self.attention=Free_attention
        self.n_head=1
        self.head_dim=self.out_dims
    def get_attention(self):
        layer=self.attention(out_dims =self.out_dims,
                              max_len=self.max_len,
                                n_head=self.n_head, 
                                head_dim=self.head_dim,
                                drop_rate=self.drop_rate, 
                                kernel_initializer=self.kernel_initializer,
                                use_bias=self.use_bias,
                                use_rotary=self.use_rotary,
                                use_Time_shift=self.use_Time_shift,
                                mask_future=self.mask,
                                )
        return layer
    def get_config(self):
        config = {
            'out_dims': self.out_dims,
            'max_len':self.max_len,
        }
        base_config = super(Free_EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Free_Encoder(Free_EncoderLayer):
    def __init__(self,n_layer,**kwargs):
        '''
         max_len:序列的最大长度
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
        super(Free_Encoder, self).__init__(**kwargs)
        self.n_layer = n_layer
        self.attention=Free_EncoderLayer
    def set_weight(self,input_shape):
        self.ls = [ self.get_attention() for _ in range(self.n_layer)]
    def call(self, xz,mask=None):
        for l in self.ls:
            xz = l(xz,mask)
        return xz       # [n, step, dim]
    def get_config(self):
        config = {
            'n_layer': self.n_layer,
        }
        base_config = super(Free_EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Free_DecoderLayer(Free_EncoderLayer):
    '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
    def set_weight(self,input_shape):
        self.ln = [LayerNormalization() for _ in range(3)] # only norm z-dim
        self.drop = keras.layers.Dropout(self.drop_rate)
        att2=self.get_attention()
        self.mask=True
        att1=self.get_attention()
        self.mh = [att1,att2]
        self.ffn = PositionWiseFFN(self.n_head*self.head_dim,
                                   self.kernel_initializer,
                                   self.use_bias,activation=self.activation)
    def call(self, inputs):
        yz, xz=inputs[0],inputs[1]
        if self.use_Time_shift:
            xz=self.time_shift(xz)
            yz=self.time_shift(yz)
        attn = self.mh[0]([yz, yz, yz])       # decoder self attention
        o1 = self.ln[0](attn+yz )
        attn = self.mh[1]([o1, xz, xz])       # decoder + encoder attention
        o2 = self.ln[1](keras.layers.Add()([attn,o1]) )
        ffn = self.drop(self.ffn(o2))
        o = self.ln[2](keras.layers.Add()([ffn ,o2]))
        return o
class Free_Decoder(Free_Encoder):
    '''
         n_head:多头注意力头个数
         head_dim：每个头的维度
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
         n_layer：一个encoder所需要堆叠的encoder_layer数
        '''
    def __init__(self,**kwargs):
        super(Free_Decoder,self).__init__(**kwargs)
        self.attention=Free_DecoderLayer
    def call(self, inputs):
        #yz代表目标，xz代表encoder的结果
        yz, xz=inputs[0],inputs[1]
        for l in self.ls:
            yz = l([yz, xz])
        return yz
