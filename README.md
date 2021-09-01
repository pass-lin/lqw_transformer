# tf_transformer
基于tf2.0实现的transformer
想要创建一个transformer encoder只需要从models中import EncodeLayer
随后
layers=EncodeLayern_head=头数, 
                 head_dim=头的维度,
                 drop_rate=drop比例, ）即可
models除了提供transformer基础架构外，还有rope位置编码，time-shift功能和AFT的自主实现。以后还会添加更多的轮子的
具体注释详见代码
如有疑问可添加笔者的QQ935499957
作者林庆文
