import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='inception-bn-28-small',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='cifar10/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str, default="0",
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=60000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=600,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load/save')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
args = parser.parse_args()


def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3,3), num_filter=20)
#     tanh1 = mx.symbol.Activation(data=conv1, act_type="relu")
#     pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
#                               kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=conv1, kernel=(3,3), num_filter=50)
#     tanh2 = mx.symbol.Activation(data=conv1, act_type="relu")
#     pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
#                               kernel=(2,2), stride=(2,2))
    
    conv3 = mx.symbol.Convolution(data=conv2, kernel=(3,3), num_filter=50)
    
    # first fullc
    flatten = mx.symbol.Flatten(data=conv3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="relu")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax', )
    return lenet



# data_shape = (784, )
# net = get_mlp()

#data_shape = (1, 28, 28)
#net = get_lenet()


# network
import importlib
net = importlib.import_module("symbol_inception-bn-28-small-j").get_symbol(2)


# data
def get_iterator(args, kv):
    data_shape = (1, 28, 28)
    train = mx.io.ImageRecordIter(
        path_imgrec = "craters.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = False,
        rand_mirror = False,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    return (train, train)

# train
train_model.fit(args, net, get_iterator)
