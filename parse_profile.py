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
def get_profile_data(config):
    network = config['network']
    batch_size = config['batch_size']
    dev = config['dev']
    dry_run = config['dry_run']
    iteration = config['iteration']
    out_dir = config['out_dir']

    # get mod
    if 'ssd' in network:
        sym, data_shape = get_ssd_symbol(network, batch_size)
    else:
        sym, data_shape = get_symbol(network, batch_size)

    interals = sym.get_internals()
    _, out_shapes, _ = interals.infer_shape(data=(batch_size,3,299,299))
    shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    # match shape with profiler result
    profile_filename = 'profile.json'
    with open(profile_filename, 'r') as f:
        profile = json.loads(f.read())['traceEvents']
    other_type = ['sgd_update', 'CopyCPU2CPU', '_zeros', 'SetValueOp', \
                  '_backward_copy', 'DeleteOperator', '_random_uniform', \
                  'WaitForVar', 'DeleteVariable', 'SetupExec', '_full', 'ResourceParallelRandomSetSeed']
    #did not consider all possibilities
    profile_list = [d for d in profile if not 'args' in d and not d['name'] in other_type]


    i = 0
    print len(profile_list)
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
    #profile position
    pos = 0


    for it in xrange(iteration):
        #forward
        for layer in topo:
            start = profile_list[pos]
            end = profile_list[pos+1]
            time = int(end['ts']) - int(start['ts'])
            layer_type = start['name']
            layer_name = layer['name']
            shape = get_input_shape(layer_name, shape_dict, topo)
            layer_config = get_layer_config(layer_name, topo)
            layer_profile.append({'iteration':it, 'layer_name':layer_name, \
            'layer_type':layer_type, 'time':time, 'shape':shape, 'config':layer_config})
            pos += 2
        #backward TODO optimize code
        for layer in topo[::-1]:
            start = profile_list[pos]
            end = profile_list[pos+1]
            time = int(end['ts']) - int(start['ts'])
            layer_type = start['name']
            layer_name = layer['name']
            shape = get_input_shape(layer_name, shape_dict, topo)
            layer_config = get_layer_config(layer_name, topo)
            layer_profile.append({'iteration':it, 'layer_name':layer_name, \
            'layer_type':layer_type, 'time':time, 'shape':shape, 'config':layer_config})
            pos += 2

    json_str = json.dumps(layer_profile, indent = 2)
    result_file_path = os.path.join(out_dir, '{}-{}.json'.format(network, batch_size))
    with open(result_file_path, 'w') as f:
        f.write(json_str)
        f.close()


def main():
    args = args_process()
    with open(args.config) as f:
        config = json.loads(f.read())
    #gen profile data
    # os.system('export MXNET_PROFILER_AUTOSTART=1')
    # os.system('export MXNET_PROFILER_MODE=1')
    # os.system('python benchmark_profile.py -c {}'.format(args.config))
    # os.system('export MXNET_PROFILER_AUTOSTART=0')
    # os.system('export MXNET_PROFILER_MODE=0')
    #parse profile data
    profile_result = get_profile_data(config)
if __name__ == '__main__':
    main()
