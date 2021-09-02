# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 19:56:18 2021

@author: Administrator
"""
import numpy as np

def topk_search(encode_vector,decode,data,start,end,maxlen=100,k=10):
    #一次topk解码
    l=len(encode_vector)
    decode_result=np.array([[259]]*l)
    n=0
    stop=np.zeros(l)*-1
    while sum(stop[:]==1)!=l:
        if n>200:
            break
        n+=1
        index=stop[:]!=1
        decoder_input=decode_result[index]
        vector=encode_vector[index]
        y=decode.predict([vector,decoder_input])
        #y=np.argmax(y,-1)
        y=y[:,-1,:]
        y=np.reshape(y,[len(y),-1])
        probility=np.sort(y,axis=-1)[:,-k:]
        probility_s=np.sum(probility,axis=-1,keepdims=True)
        probility=probility/probility_s
        y_sort=y.argsort(-1)[:,-k:]
        y=[np.random.choice(y_sort[i],p=probility[i]) for i in range(len(y_sort))]
        y=np.reshape(y,[-1,1])
        stop[index]=y[:,-1]
        t=np.zeros([l])
        t[index]=y[:,-1]
        t=np.reshape(t,[l,1])
        decode_result=np.concatenate([decode_result,t],-1)
    return decode_result,stop
def topk_decode(encoder,decode,data,start,end,maxlen=100,k=10):
    #topk解码使用这个
    encode_vector=encoder.predict(data)
    l=len(encode_vector)
    decode_result=np.array([[start]]*l)
    stop=np.zeros(l)*-1
    while sum(stop[:]==end)!=len(stop):
        #部分停不下来的要重新解码到可以为止，所以要while循环
        newdecoder,newtop=topk_search(encode_vector,decode,data[stop[:]!=end],start,end,maxlen,k)
        t=np.zeros([len(decode_result),np.shape(newdecoder)[-1]])*end
        decode_result=np.concatenate([decode_result,t],-1)
        decode_result[stop[:]!=end]=newdecoder
        stop[stop[:]!=end]=newtop
    return decode_result
def greedy_search(encoder,decode,data,start,end,maxlen=100,k=10):
    #贪婪解码，encoder是你的编码器，decode是解码器，data是数据
    #start代表你词典中开始符的位置，end为结束符
    #k是topk解码的所用的
    encode_vector=encoder.predict(data)
    l=len(encode_vector)
    decode_result=np.array([[start]]*l)
    n=0
    stop=np.zeros(l)*-1
    while sum(stop[:]==1)!=l:
        if n>maxlen:
            break
        n+=1
        index=stop[:]!=1
        decoder_input=decode_result[index]
        vector=encode_vector[index]
        y=decode.predict([vector,decoder_input])
        y=np.argmax(y,-1)
        stop[index]=y[:,-1]
        t=np.zeros([l])*end
        t[index]=y[:,-1]
        t=np.reshape(t,[l,1])
        decode_result=np.concatenate([decode_result,t],-1)
    #部分无法停下来的用topk
    newdecoder=topk_decode(encoder,decode,data[stop[:]!=end],start,end,maxlen,k)
    t=np.zeros([len(decode_result),np.shape(newdecoder)[-1]])*end
    decode_result=np.concatenate([decode_result,t],-1)
    decode_result[stop[:]!=end]=newdecoder
    return decode_result