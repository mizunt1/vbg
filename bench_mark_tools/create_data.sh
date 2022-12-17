#!/bin/bash
base_dir=/network/scratch/m/mizu.nishikawa-toomey/gflowdag
mkdir ${base_dir}/results1
mkdir ${base_dir}/results2
for n in {0..20}; do mkdir ${base_dir}/results1/seed${n} ;done 
for n in {0..20}; do mkdir ${base_dir}/results1/seed${n}/gflowdag ;done 

for n in {0..20}; do mkdir ${base_dir}/results2/seed${n} ;done 
for n in {0..20}; do mkdir ${base_dir}/results2/seed${n}/gflowdag ;done 

python bench_mark_tools/save_data_graph_full_posterior.py
python bench_mark_tools/save_data_and_graph.py

