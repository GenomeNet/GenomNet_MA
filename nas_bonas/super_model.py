

from generalNAS_tools.operations_14_9 import *
from generalNAS_tools.genotypes import PRIMITIVES_cnn, PRIMITIVES_rnn, rnn_steps, CONCAT
from opendomain_utils.training_utils import mask2d

from opendomain_utils import genotypes
import numpy as np
from torch.autograd import Variable

from darts_tools.comp_aux import get_state_ind, get_w_pos

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INITRANGE = 0.04

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math




class MixedOp(nn.Module):
    '''mask: op'''
    
    # mask = supernet_mask[1]
    def __init__(self, C, stride, mask_cnn):
        super(MixedOp, self).__init__()
        self.stride = stride
        self._ops = nn.ModuleList()
        # mask = supernet_mask[0]
        # mask_old = mask_cnn
        # mask_cnn = mask_old[0]
        mask_1 = np.nonzero(mask_cnn)[0] # 
      
        self._super_mask = mask_1 
        if len(mask_1) != 0:
            for selected in np.nditer(mask_1): # iterates over the operations         
                primitive = PRIMITIVES_cnn[selected] 
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
                self._ops.append(op)

    
    def forward(self, x, mask_cnn):
        if (mask_cnn==0).all(): 
            # falls normal cell
            if self.stride == 1:
                # return x.mul(0.)
                return torch.zeros_like(x)
            return torch.zeros_like(x[:,:,::self.stride]) 
        else: 
            result = 0
            mask_2 = np.nonzero(mask_cnn)[0] 
            if len(mask_2) != 0:
                for selected in np.nditer(mask_2): 
                    # pos = np.where(_super_mask==selected)[0][0] 
                    # print(pos)

                    pos = np.where(self._super_mask==selected)[0][0] 
              
                    result += self._ops[pos](x)
                    
            return result


class CNN_Cell_search(nn.Module):
    '''mask: 14 * 8'''

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, mask_cnn, reduction_high):
                       # steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, mask_cnn, reduction_high = 4, 4, 8, 8, 8, False, False, mask_cnn, False
        super(CNN_Cell_search, self).__init__()
        
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine=False)
            #self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        cnt = 0
        
        for i in range(self._steps):
            for j in range(2 + i):
                
                stride = 2 if reduction and j < 2 else 1
                    
                if (reduction==False) or j >= 2:
                    stride=1
                
                if reduction and j < 2:
                    stride=2
                    
                if reduction_high and j < 2:
                    stride=3
                
                op = MixedOp(C, stride, mask_cnn[cnt])
                
                self._ops.append(op) 
                
                cnt += 1
                

    def forward(self, s0, s1, mask_cnn):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j].forward(h, mask_cnn[offset + j]) for j, h in enumerate(states))
        
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)



# for evaluation
class DARTSCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype

    steps = len(self.genotype.rnn) if self.genotype is not None else rnn_steps
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) 
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps) 
    ])
   
  def forward(self, inputs, hidden, rnn_mask):
  
    T, B = inputs.size(0), inputs.size(1) # 10,2

    if self.training:
        
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx) 

      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)

    else:
        
      x_mask = h_mask = None
      
    hidden = hidden[0]

    hiddens = []
    for t in range(T):
 
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask, rnn_mask)

      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):

    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1) # entlang der channels zusammenfügen
    else:
      #print(x.shape)
      #print(h_prev.shape)
      xh_prev = torch.cat([x, h_prev], dim=-1)
      
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1) 
    # c0, h0 = torch.split(xh_prev.mm(_W0), nhid, dim=-1) 


    c0 = c0.sigmoid() 
    h0 = h0.tanh() 
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def _get_activation(self, name):
    if name == 'tanh':
      f = torch.tanh
    elif name == 'relu':
      f = torch.relu
    elif name == 'sigmoid':
      f = torch.sigmoid
    elif name == 'identity':
      f = lambda x: x
    else:
      raise NotImplementedError
    return f

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    states = [s0]
    for i, (name, pred) in enumerate(self.genotype.rnn):
      s_prev = states[pred]
      if self.training:
        ch = (s_prev * h_mask).mm(self._Ws[i])
      else:
        ch = s_prev.mm(self._Ws[i])
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      fn = self._get_activation(name)
      h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
    return output




# C, num_classes, layers, criterion, steps, multiplier, stem_multiplier = 16, 4, 3, nn.CrossEntropyLoss().to(device), 4, 4, 3 
# _C, _num_classes, _layers, _criterion, _steps, _multiplier, _stem_multiplier = 16, 4, 3, nn.CrossEntropyLoss().to(device), 4, 4, 3 

# x = torch.rand(2,4,10)
# x = torch.rand(10,2,256) # nach CNN_cells
# hidden = torch.rand(1,2,256)
# hidden = hidden[0] # [1,2,256] 
# ninp, nhid, nhidlast = 256, 256, 256


class RNNModel(nn.Module):

    #self.seq_len, self.dropouth, self.dropoutx,
    #                          self.init_channels, self.num_classes, self.layers, self.steps, multiplier, stem_multiplier,  
    #                          True, 0.2, None, self.task, 
    #                          switches_normal_cnn, switches_reduce_cnn, switches_rnn, 0.0
    def __init__(self, seq_len, 
                 dropouth=0.5, dropoutx=0.5,
                 C=8, num_classes=4, layers=3, steps=4, multiplier=4, stem_multiplier=3,
                 search=True, drop_path_prob=0.2, genotype=None, task=None, mask=None):
                
        super(RNNModel, self).__init__()
    
            
        self.mask_cnn = mask[0]
        self.mask_rnn = mask[1]
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        #self.p = p
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        
        #if search == True:
        
        #    self.switches_normal = switches_normal
        #    self.switches_reduce = switches_reduce
        #    self.switches_rnn = switches_rnn
        #    self.num_ops_cnn = sum(switches_normal[0])
        #    self.num_ops_rnn = sum(switches_rnn[0])

    
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv1d(4, C_curr, 13, padding=6, bias=False),
            nn.BatchNorm1d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        layer_list = []
        
        for i in range(layers):
            layer_list.append(i)
        
        normal_cells = layer_list[::3]
        
        num_neurons = seq_len # =1000
        
        for i in range(layers):
       
            if i not in normal_cells:
                
                C_curr *= 2
                reduction = True
                
                if (i==5):
                    reduction_high=True
                    num_neurons = round(num_neurons/3)
                    #stride=3
                else:
                    reduction_high=False
                    num_neurons = round(num_neurons/2) #int(math.ceil(num_neurons/2))
                    
                if search == True:
                    # switches = self.switches_reduce
                    # steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, mask_cnn, reduction_high
                    cell = CNN_Cell_search(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.mask_cnn, reduction_high)
                    # steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches, self.p, reduction_high
                else:
                    cell = cnn_eval.CNN_Cell_eval(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, reduction_high)

            else:
                reduction = reduction_high = False
                # reduction = False
                if search == True:
                    # switches = self.switches_normal
                    cell = CNN_Cell_search(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.mask_cnn, reduction_high) 
                else:
                    cell = cnn_eval.CNN_Cell_eval(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, reduction_high) 
              
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.drop_path_prob = drop_path_prob
        
        out_channels = C_curr*steps
        
        # num_neurons = math.ceil(math.ceil(seq_len / len([layers//3, 2*layers//3])) / 2)
        
        ninp, nhid, nhidlast = out_channels, out_channels, out_channels
    
    
        # x = torch.rand(10,2,256)
        # hidden = hidden[0] # [1,2,256]
            
        assert ninp == nhid == nhidlast
        # again, we have different rnn cells for search and for evaluation
        if search == False: 
            # rnn cell for evaluation of final architecture
            assert genotype is not None
            cell_cls = DARTSCell
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)] # 
            # rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]

        else:
            # run search
            assert genotype is None
            from BONAS_search_space import model_search
            cell_cls = model_search.DARTSCellSearch
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx)]
        
        
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(num_neurons*out_channels, 925) # because we have flattened 
        
        self.classifier = nn.Linear(925, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout_lin = nn.Dropout(p=0.1)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        
        self.cell_cls = cell_cls
        
        if task == "TF_bindings":
            
            self.final = nn.Sigmoid()
            
        else:
            
            self.final = nn.Identity()

        
    def init_weights(self):
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)  

    def forward(self, input, hidden, mask, return_h=False):
        
#        print(mask[0])
        s0 = s1 = self.stem(input)
        
        for i, cell in enumerate(self.cells):        
            
            s0, s1 = s1, cell.forward(s0, s1, mask[0])
       
        batch_size = s1.size(0) 

        out_channels = s1.size(1)
        num_neurons = s1.size(2)
        
        #CNN expects [batchsize, input_channels, signal_length]
        # RHN expect [seq_len, batch_size, features]
        
        x = s1.permute(2,0,1)
        # emb = torch.reshape(s1, (num_neurons,batch_size,out_channels)) 

        raw_output = x
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns): 
            
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l], mask[1])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            
        hidden = new_hidden

        output = raw_output
        
        outputs.append(output)
        
        output = output.permute(1,0,2)

        
        # flatten the RNN output
        x = torch.flatten(output, start_dim= 1) 
        
        # dropout layer
        x = self.dropout_lin(x)
        
        # linear layer
        x = self.decoder(x) 
        
        # dropout layer
        #x = self.dropout_lin(x)
        
        x = self.relu(x)
        
        # print('first')
        # print(x)
        
        # linear layer
        x = self.classifier(x)

        logit = self.final(x)

        if return_h:
            return logit, hidden, raw_outputs, outputs
        
        return logit, hidden 
     

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self(input, hidden, return_h=False) 
          
        criterion = nn.CrossEntropyLoss()
        loss = criterion(log_prob, target)
        
        return loss, hidden_next
  
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [Variable(weight.new(1, bsz, self.nhid).zero_())]



# genotype = genotypes[0]
def geno2mask(genotype):
    des = -1
    #mask = np.zeros([14+36,8+4])
    
    cnn_mask = np.zeros([14,9])
    rnn_mask = np.zeros([36,5])

    op_names_cnn, indices = zip(*genotype.normal)
    
    for cnt, (name, index) in enumerate(zip(op_names_cnn, indices)):
     
        if cnt % 2 == 0: 
       
            des += 1 
            total_state = sum(i + 2 for i in range(des)) 
           
        op_idx = PRIMITIVES_cnn.index(name) 
        node_idx = index + total_state 
        
        cnn_mask[node_idx, op_idx] = 1 
      
    op_names_rnn, indices = zip(*genotype.rnn)

    aux = 0
    for cnt, (name, index) in enumerate(zip(op_names_rnn, indices)):
   
       
        node_idx = aux + index
      
        op_idx = PRIMITIVES_rnn.index(name) 
    
        rnn_mask[node_idx, op_idx] = 1 
        
        aux += cnt +1
        
    return cnn_mask, rnn_mask

# subnet_mask = geno2mask(genotype)

def merge(cnn_mask, rnn_mask):
    supernet_mask_cnn = np.zeros((14, 9))
    supernet_mask_rnn = np.zeros((36,5))
    for mask in cnn_mask: 
        supernet_mask_cnn = mask + supernet_mask_cnn
        
    for mask in rnn_mask: 
        supernet_mask_rnn = mask + supernet_mask_rnn
    return supernet_mask_cnn, supernet_mask_rnn


if __name__ == "__main__":
    num_ops = 8
    steps = 4
    arch = "BONAS"
    genotype = eval("genotypes.%s" % arch)
    mask = geno2mask(genotype)

    print(str(mask))
    print(hash(mask.tostring()))
    k = sum(1 for i in range(steps) for n in range(2 + i))
    mask = torch.ones(k, num_ops)

    '''---------------------------------------------'''
    mask = np.array([[0, 1, 0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1]]
                    )
    # print(mask)
    mix_op = MixedOp(C=16, stride=1, mask=mask[0])
    cell = Cell(steps, multiplier=4, C_prev_prev=16, C_prev=16, C=16, reduction=False, reduction_prev=False, mask=mask)
