# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Benchmark the scoring performance on various CNNs
"""
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

def score(network, dev, batch_size, num_batches):
    # get mod
    if 'ssd' in network:
        sym, data_shape = get_ssd_symbol(network, batch_size)
    else:
        sym, data_shape = get_symbol(network, batch_size)

    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)


def run_profiler(sym, data_shape, dev, iteration, dry_run):
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

    #config profiler
    profile_filename = 'profile.json'
    profiler.profiler_set_config(filename=profile_filename)
    profiler.profiler_set_state('run')

    #real run
    for i in xrange(iteration):
        mod.forward(batch, is_train=True)
        mod.backward()
        mod.update()
        for output in mod.get_outputs():
            output.wait_to_read()
    profiler.profiler_set_state('stop')

    #dump profile
    # profiler.dump_profile()


def get_profile_data(network, dev, batch_size, iteration, dry_run):

    # get mod
    if 'ssd' in network:
        sym, data_shape = get_ssd_symbol(network, batch_size)
    else:
        sym, data_shape = get_symbol(network, batch_size)

    run_profiler(sym, data_shape, dev, iteration, dry_run)
    print data_shape
    
    # get input shape config
    interals = sym.get_internals()
    _, out_shapes, _ = interals.infer_shape(data=(batch_size,3,299,299))
    shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    print shape_dict
    #match shape with profiler result
    profile_filename = 'profile.json'
    with open(profile_filename, 'r') as f:
        profile = json.loads(f.read())['traceEvents']
    profile_list = [d for d in profile if not 'args' in d]
    i = 0

    #TODO find other way to do this!
    layer_type_count = {}
    layer_profile = []
    topo = [d for d in json.loads(sym.tojson())['nodes'] if d['op'] != 'null']

    def get_input_shape(layer_name, shape_dict, topo):
        for prelayer, layer in zip(topo, topo[1:]):
            if layer['name'] == layer_name:
                return shape_dict[prelayer['name']+'_output']
        if layer_name == topo[0]['name']:
            #return the data shape
            return (batch_size, 3, 299, 299)

    def get_layer_config(layer_name, topo):
        for layer in topo:
            if layer['name'] == layer_name:
                if 'attrs' in layer:
                    return layer['attrs']
                else:
                    return None

    while i < len(profile_list)-1:
        start = profile_list[i]
        end = profile_list[i+1]
        time = int(end['ts']) - int(start['ts'])
        layer_type = start['name']
        if i/2 > len(topo) - 1:
            layer_name = topo[-(i/2 - len(topo) + 1)]['name']
        else:
            layer_name = topo[i/2]['name']
        shape = get_input_shape(layer_name, shape_dict, topo)
        layer_config = get_layer_config(layer_name, topo)
        if not layer_type in layer_type_count:
            layer_type_count[layer_type] = 0
        else:
            layer_type_count[layer_type] += 1

        layer_profile.append({'layer_name':layer_name,'layer_type':layer_type, 'time':time, 'shape':shape, 'config':layer_config})
        i += 2

    return layer_profile

def run_profile_test(config):
    
    networks = config['network']
    batch_size = config['batch_size']
    dev = config['dev']
    dry_run = config['dry_run']
    iteration = config['iteration']
    level = config['level']
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

    for net in networks:
        logging.info('network: {}'.format(net))
        for d in dev_list:
            logging.info('device: {}'.format(d))
            for b in batch_size:
                logging.info('batch size {}, dry_run: {}, level {}'.format(b, dry_run, level))
                file_name = '{}-{}-{}-{}.json'.format(net, d, b, level)
                file_path = os.path.join(out_dir, file_name)
                profile_result = get_profile_data(net, d, b, iteration, dry_run)
                with open(file_path, 'w') as f:
                    json_str = json.dumps(profile_result, indent=2)
                    f.write(json_str)
                    f.close()
                logging.info('test result saved at {}'.format(file_path))
    

def main():
    args = args_process()
    with open(args.config) as f:
        config = json.loads(f.read())

    logging.info(config)
    test_type_option = {'profile':run_profile_test, 'score':score}
    #run test
    test_type_option[config['test']](config)
 

if __name__ == '__main__':
    # networks = ['ssd.VGG16_reduced', 'mobilenet', 'alexnet', 'vgg-16', 'inception-bn', 'inception-v3', 'resnet-50', 'resnet-152']
    # networks = ['ssd.vgg16_reduced', 'ssd.inceptionv3']
    # batch_sizes = [1, 2, 4, 8, 16, 32]

    main()

