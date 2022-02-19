#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:14:42 2021

@author: amadeu
"""

# für seq_len 200

import torch
import torch.nn as nn
import math

########### exports:
# OPS, NorConv, ReLUConvBN, DilConv, SepConv, Identity, Zero, FactorizedReduce


#op = OPS[primitive](C, stride, False)

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  #'avg_pool_5' : lambda C, stride, affine: nn.AvgPool1d(5, stride=stride, padding=2, count_include_pad=False), 
  #'max_pool_5' : lambda C, stride, affine: nn.MaxPool1d(5, stride=stride, padding=2),
  'avg_pool_5' : lambda C, stride, affine: nn.AvgPool1d(5, stride=stride, padding=2, count_include_pad=False), # 9,2
  'max_pool_5' : lambda C, stride, affine: nn.MaxPool1d(5, stride=stride, padding=2), # 9,2
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, stride=stride, affine=affine),
  'sep_conv_15' : lambda C, stride, affine: SepConv(C, C, 15, stride, 7, 7, affine=affine), 
  'sep_conv_9' : lambda C, stride, affine: SepConv(C, C, 9, stride, 4, 4, affine=affine),
  #'sep_conv_13' : lambda C, stride, affine: SepConv(C, C, 9, stride, 4, affine=affine), 
  'dil_conv_15' : lambda C, stride, affine: DilConv(C, C, 15, stride, 14, 2, affine=affine),
  'dil_conv_9' : lambda C, stride, affine: DilConv(C, C, 9, stride, 8, 2, affine=affine),
  'conv_15' : lambda C, stride, affine: NorConv(C, C, 15, stride, affine=affine),
  }

# fc = torch.nn.Identity(2,2)
#for name, param in fc.named_parameters():
#        print(name)
#        print(param.shape)
        #if 'stem' or 'preprocess' or 'conv' or 'bn' or 'fc' in name:
#        if 'rnns' in name:
            #print(name)
#            rhn.append(param)
        #elif 'decoder' in name:
#        else:
            #print(name)
#            conv.append(param)

# to check for the pooling layers

#seq_len = 500 + 2*7
#kernelsize = 9
#start=0
#end = kernelsize-1
#receptive_fields = []
#while start < seq_len:
#    receptive_fields.append([start,end])
#    start += 3
#    end += 3
    
#x = torch.rand(2,4,42)
#conv = nn.Conv1d(4, 4, 15, stride=1, padding=7, bias=False)
#x = conv(x)
#x.shape # 201 falls kernel=14
#conv = nn.Conv1d(4, 4, 15, stride=1, padding=7, bias=False)
#x = conv(x)
#x.shape # 200



class NorConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, affine=True):
    super(NorConv, self).__init__()
    self.op = nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C_in, C_in, kernel_size, stride=stride, padding=7, bias=False),
    nn.BatchNorm1d(C_in, affine=affine),
    nn.ReLU(inplace=False),
    nn.Conv1d(C_in, C_out, kernel_size, stride=1, padding=7, bias=False),
    nn.BatchNorm1d(C_out, affine=affine),
    )

  def forward(self, x):
    return self.op(x)




#x = torch.rand(2, 4, 125)
#C = 4
#stride=3
#kernel_size = 15
#conv = NorConv(C,C,kernel_size,stride)

#x = conv(x)
#x.shape # ergibt 37


# je höher kernelsize desto höher muss padding sein 
# erstmal abchecken wie viel bei ihm reduziert wird: immer um hälfte!!!
#x = torch.rand(2, 100, 200)

#max_nor = nn.MaxPool1d(5, stride=1, padding=2)
#max_red = nn.MaxPool1d(5, stride=2, padding=2)
#x = max_nor(x)
#x = max_red(x)
#x = torch.rand(2, 4, 1000)
#avg_nor = nn.AvgPool1d(5, stride=1, padding=2, count_include_pad=False)
#avg_red = nn.AvgPool1d(5, stride=2, padding=2, count_include_pad=False)
#x = avg_nor(x)
#x = avg_red(x)

#x.shape


#num_motifs = 100

#reconv = ReLUConvBN(100,100,1,1, 0)
#x = torch.rand(2, 100, 150)
#C = 100
#stride=1
#reconv = ReLUConvBN(C,C,1,1,0,affine=False)
#x = reconv(x)
#x.shape


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm1d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)




# to check for the dilconv layers
#seq_len = 200 + 13 # d.h. 207tes element ist letztes seqeuenz element und von 207 bis 214 sind nur 0en (ebenso von 0 bis 7 nur 0en)
#seq_len = 125 + 14*2
#kernelsize = 15
#start=0
#end = (kernelsize-1)+(kernelsize-1)
#receptive_fields = []
#while start < seq_len:
#    receptive_fields.append([start,end])
#    start += 3
#    end += 3


#x = torch.rand(2, 4, 250)
#C = 4
#stride=3
#conv = DilConv(C, C, 15, stride, 14, 2) # ergibt 42 falls stride=3 bei kernel=15 aber auch kernel 
# 9, 8, 2 # ergibt 38
# 14, 13, 2 # ergibt 38
#x = conv(x)
#x.shape



class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)



#x = torch.rand(2, 4, 250)
#C = 4
#stride=3
#conv = SepConv(C, C, 15, stride, 7, 7) # 9, 4, 4; 14, 7, 6,
#x = conv(x)
#x.shape # 42 falls stride 3

# 9, 4, 4 ergibt 38 Neuronen
# 14, 7, 6 ergibt 37 Neuronen

class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding_1, padding_2, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding_1, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding_2, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


#x = torch.rand(2,100,150)
#stride=1
#iden = Identity()
#out = iden(x)
#out.shape


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

'''
class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)
'''

#x = torch.rand(2,4,125)
#stride=3
#zero = Zero(stride)
#x = zero(x) # 63 also aufgerundet
#x.shape
# ergibt 38

class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride].mul(0.)


#class Zero(nn.Module):

#  def __init__(self, stride):
#    super(Zero, self).__init__()
#    self.stride = stride
#  def forward(self, x):
#    n, c, h = x.size()
#    h /= self.stride # divides h with stride and rounds result off (abrunden)
#    h = math.ceil(h)
#    if x.is_cuda:
#      with torch.cuda.device(x.get_device()):
#        padding = torch.cuda.FloatTensor(n, c, h).fill_(0)
#    else:
#      padding = torch.FloatTensor(n, c, h).fill_(0)
#    return padding    



#x = torch.rand(2,4,125)
#stride=3 # weil bei stride 1 wäre es Identity() funktion 2 weiter hoch
#ident = FactorizedReduce(4, 4)
#x = ident(x)
#x.shape
# 38 Neuronen


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, stride, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False) # jede output channel wird durch 2 geteilt, aber da am ende zusammengefügt wird, hat output C_out als output channels
    self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False) 
    self.bn = nn.BatchNorm1d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    #print(self.conv_1(x).shape) 
    #print(self.conv_2(x[:,:,1:]).shape)

    out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1) # anzahl batches und anzahl channels soll gleich bleiben , aber im 
    # 2d fall soll die von beiden neuronen dimensinonen die erste spalte bzw. erste zeile gekickt werden
    # an dimension 1 zusammenfügten, also anhand von channels (weil batches ist dimension 0) zusammenfügen
    #out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1) 
    out = self.bn(out)
    return out



