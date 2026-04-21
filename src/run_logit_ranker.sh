#!/bin/bash

for arg in 1  5; do
    for lrg in 18 20 22 24 ; do
        python src/run_logit_ranker.py -i music -m qwen3b -l "$lrg" -a "$arg" -d gender -bs steered
    done
done
