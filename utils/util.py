# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

from torch.autograd import Variable
import torch



def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
