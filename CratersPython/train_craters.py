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



parser.add_argument('--fold', type=int, default='1')
parser.add_argument('--region', type=str, default='East')




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
        path_imgrec = "data/" + args.region + "-Fold" + `args.fold` +  "-train.rec",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = False,
        rand_mirror = False,
        shuffle = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank
    )
    
    test = mx.io.ImageRecordIter(
        path_imgrec = "data/" + args.region + "-Fold" + `args.fold` +  "-test.rec",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = False,
        rand_mirror = False,
        shuffle = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank
    )
    
    return (train, test)

# train
train_model.fit(args, net, get_iterator)
