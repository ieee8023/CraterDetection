
# coding: utf-8

# In[65]:

import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
# parser.add_argument('--network', type=str, default='inception-bn-28-small-j',help = 'the cnn to use')
# parser.add_argument('--data-dir', type=str, default='cifar10/',
#                     help='the input data directory')
# parser.add_argument('--gpus', type=str, default="0",
#                     help='the gpus will be used, e.g "0,1,2,3"')
# parser.add_argument('--num-examples', type=int, default=60000,
#                     help='the number of training examples')
# parser.add_argument('--batch-size', type=int, default=600,
#                     help='the batch size')
# parser.add_argument('--lr', type=float, default=.05,
#                     help='the initial learning rate')
# parser.add_argument('--lr-factor', type=float, default=1,
#                     help='times the lr with a factor for every lr-factor-epoch epoch')
# parser.add_argument('--lr-factor-epoch', type=float, default=1,
#                     help='the number of epoch to factor the lr, could be .5')
# parser.add_argument('--model-prefix', type=str,
#                     help='the prefix of the model to load/save')
# parser.add_argument('--num-epochs', type=int, default=100,
#                     help='the number of training epochs')
# parser.add_argument('--load-epoch', type=int,
#                     help="load the model on an epoch using the model-prefix")
# parser.add_argument('--kv-store', type=str, default='local',
#                     help='the kvstore type')
# args = parser.parse_args()


# In[70]:


parser = argparse.ArgumentParser(description='Hello')
parser.add_argument('--fold', type=int, default='1')

args = parser.parse_args()


# network
import importlib
net = importlib.import_module("symbol_inception-bn-28-small-j").get_symbol(2)


data_shape = (1, 28, 28)
# data
train = mx.io.ImageRecordIter(
    path_imgrec = "East-Fold" + `args.fold` +  "-train.rec",
    data_shape  = data_shape,
    batch_size  = 100,
    rand_crop   = False,
    rand_mirror = False,
    shuffle = True
)

test = mx.io.ImageRecordIter(
    path_imgrec = "East-Fold" + `args.fold` +  "-test.rec",
    data_shape  = data_shape,
    batch_size  = 100,
    rand_crop   = False,
    rand_mirror = False,
    shuffle = True
)



# train
devs = mx.gpu(int(0)) #mx.cpu() 
lr = 0.05
lr_factor = 1
lr_factor_epoch = 1
num_epochs = 50
batch_size = 100
num_examples = 1000

epoch_size = num_examples / batch_size

model_args = {}    

if lr_factor < 1:
    model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
        step = max(int(epoch_size * lr_factor_epoch), 1),
        factor = lr_factor)



# In[83]:

model = mx.model.FeedForward(
    ctx                = devs,
    symbol             = net,
    num_epoch          = num_epochs,
    learning_rate      = lr,
    momentum           = 0.9,
    wd                 = 0.00001,
    initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
    **model_args)


# In[ ]:

model.fit(
    X                  = train,
    eval_data          = test,
    #kvstore            = kv,
    batch_end_callback = mx.callback.Speedometer(batch_size, 50),
    #epoch_end_callback = checkpoint
    eval_metric = 'f1'
)


# In[ ]:




