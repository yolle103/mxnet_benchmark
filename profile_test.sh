#!/bin/bash
config=$1
export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0
export MXNET_PROFILER_AUTOSTART=1
python run_profiler.py -c $1
export MXNET_PROFILER_AUTOSTART=0                      
python parse_profile.py -c $1                        

