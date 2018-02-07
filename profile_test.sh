#!/bin/bash
config=$1
export MXNET_PROFILER_AUTOSTART=1
export MXNET_PROFILER_MODE=1
python run_profiler.py -c $1
export MXNET_PROFILER_AUTOSTART=0                      
export MXNET_PROFILER_MODE=0
python parse_profile.py -c $1                        

