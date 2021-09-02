# lqw_transformer
笔者基于tf.keras所实现的transformer及其变体

以后笔者对什么轮子感兴趣复现了其keras版也会在该库更新。

###### layers.py

layers是实现基础层的实现，当前实现了transformer block的各个层，AFT以及Synthesizer_R/MLP_Mixer。

位置编码笔者实现了rope位置编码，同时基于bert4keras.layers源码修改了PositionEmbedding和SinusoidalPositionEmbedding。

还包括trick Time_shift，参考博客https://zhuanlan.zhihu.com/p/399480671，

各层详细问文档如下

```python
def set_gelu()#该函数可以让你的keras得到gelu激励函数

class LayerNormalization(keras.layers.Layer)
class PositionEmbedding(keras.layers.Layer)
class SinusoidalPositionEmbedding(keras.layers.Layer)
#这三层来自于bert4keras.layers的实现，其中PositionEmbedding和#SinusoidalPositionEmbedding笔者做了部分修改。但是使用了bert4keras api一致
#LayerNormalization常规用法
x=LayerNormalization()(x)

class MultiHead(keras.layers.Layer)#参考了莫烦python的实现，api如下
        '''
         n_head:多头注意力头个数，必须输入
         head_dim：每个头的维度，必须输入
         drop_rate：dropout比例
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
         use_rotary 是否使用旋转位置编码
         use_Time_shift_only:启用Time_shift，参考博客https://zhuanlan.zhihu.com/p/399480671
		 mask_future:是否mask掉未来信息
		 除此之外害提供了mask的输入api，读者可自行输入mask来自定义你的特效
		 使用方法为
		 x=MultiHead(……)([q,k,v],mask)
        '''
   class AFT_full(MultiHead)
  #使用了AFT的点击注意力层， MultiHead api中除use_rotary n_head head_dim均可用
    #out_dims 输出维度，必须输入
    #max_len 文本最大长度，必须输入
   class Synthesizer_R(AFT_full)
   #继承自 AFT_full，必须输入和AFT一致，但是输入n_head大于1时回采取多头的计算方法（不建议使用，因为没啥必要）
    #train_able R矩阵会被更新
    #等价于MLP_Mixer
    
```

###### models.py

该文件是笔者将上述层封装起来的结果，提供了两个级别的封装。

Encoderlayer/Dencoderlayer级别就是一个transformer block，其输入与其对应的注意力机制输入api是一样的。但是多了一个FFN所使用的激励函数 activation

Encoderr 和 Decoder就是自定义有几个transformer block，因此其输入在Encoderlayer/Dencoderlayer的基础上加入了n_layer：即有transformer block数量



注意Synthesizer_DecoderLayer/Synthesizer_Decoder 中s_n_head表示decoder自身计算Synthesizer的头数（不建议使用，因为没有必要）。n_head表示decoder和encoder做多头注意力的头数。



```python
#这是基于roformer的seq2seq模型实现，用于参考
inputs_en=keras.layers.Input([None])
inputs_cn=keras.layers.Input([None])
inputs=keras.layers.Embedding(input_dim=en_vocab_len,
               output_dim=256,mask_zero=True)(inputs_en)
inputs=LayerNormalization()(inputs)
inputs=keras.layers.Dropout(drop_rate)(inputs)
decoder=keras.layers.Embedding(input_dim=en_vocab_len,
               output_dim=256,mask_zero=True)(inputs_cn)
decoder=LayerNormalization()(decoder)
decoder=keras.layers.Dropout(drop_rate)(decoder)

decoder=keras.layers.Embedding(input_dim=en_vocab_len,
               output_dim=256,mask_zero=True)(inputs_cn)
output=Encoder(n_layer=1,n_head=4,head_dim=64,drop_rate=drop_rate,use_rotary=True)(inputs)
output=Decoder(n_layer=1,n_head=4,head_dim=64,drop_rate=drop_rate,use_rotary=True)([decoder,output])
output=keras.layers.Dense(cn_vocab_len,activation=keras.activations.softmax)(output)
model=keras.models.Model([inputs_en,inputs_cn],output)
```

有bug可以通过邮箱2022845866@qq.com或者对应的qq联系我
