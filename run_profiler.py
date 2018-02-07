from common import find_mxnet
from common.util import get_gpus
import mxnet as mx
from importlib import import_module
import logging
import time
import numpy as np
import symbol_factory
from mxnet import profiler
import json
import argparse
import os
import shutil
from multiprocessing import Process
logging.basicConfig(level=logging.DEBUG)


def args_process():
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--config', '-c',  help='config file for runing benchmark')
    args = arg_parser.parse_args()
    return args

def get_symbol(network, batch_size):
    image_shape = (3,299,299) if network == 'inception-v3' or 'mobilenet' else (3,224,224)
    num_layers = 0
    if 'resnet' in network:
        num_layers = int(network.split('-')[1])
        network = 'resnet'
    if 'vgg' in network:
        num_layers = int(network.split('-')[1])
        network = 'vgg'
    net = import_module('symbols.'+network)
    sym = net.get_symbol(num_classes = 1000,
                         image_shape = ','.join([str(i) for i in image_shape]),
                         num_layers  = num_layers)
    return (sym, [('data', (batch_size,)+image_shape)])

def get_ssd_symbol(network, batch_size):
    image_shape = (3,299,299)
    if 'resnet' in network:
        image_shape = (3, 224, 224)
    base_name = network.split('.')[-1]
    sym = symbol_factory.get_symbol(network, 20)
    return (sym, [('data', (batch_size,)+image_shape)])

def run_profiler(network, batch_size, dev, iteration, dry_run):
    # get mod
    if 'ssd' in network:
        sym, data_shape = get_ssd_symbol(network, batch_size)
    else:
        sym, data_shape = get_symbol(network, batch_size)
    print data_shape
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = True,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(optimizer='ccsgd',
                       optimizer_params={
                            'learning_rate': 0.0001,
                            'momentum': 0.0,
                            'wd': 0.0
                        })
    # gen data label
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    # label = [mx.nd.array(np.random.randint(1, 100, size=shape), ctx=ctx) for _, shape in mod.label_shapes]
    batch = mx.io.DataBatch(data, [])


    # dry run
    for i in xrange(dry_run):
        mod.forward(batch, is_train=True)
        mod.backward()
        for output in mod.get_outputs():
            output.wait_to_read()
        mod.update()

    #real run
    for i in xrange(iteration):
        mod.forward(batch, is_train=True)
        mod.backward()
        for output in mod.get_outputs():
            output.wait_to_read()
        mod.update()

def run_profile_test(config):
    network = config['network']
    batch_size = config['batch_size']
    dev = config['dev']
    dry_run = config['dry_run']
    iteration = config['iteration']
    out_dir = config['out_dir']

    #config dev
    if dev == 'gpu':
        dev_list = [mx.gpu(0)] if len(get_gpus()) > 0 else []
    elif dev == 'cpu':
        dev_list = [mx.cpu()]
    else:
        logging.error('no valid device')


    #clean and create out_dir
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)
    logging.info('network: {} dev {}'.format(network, dev))
    logging.info('batch size {}, dry_run: {}, iteration {}'.format(batch_size, dry_run, iteration))
    run_profiler(network, batch_size, dev_list[0], iteration, dry_run)

def main():
    args = args_process()
    with open(args.config) as f:
        config = json.loads(f.read())

    test_type_option = {'profile':run_profile_test}
    #run test
    test_type_option[config['test']](config)


if __name__ == '__main__':
    main()

