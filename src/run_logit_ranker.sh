#!/bin/bash

for arg in 1 2 5; do
    for lrg in 20 22 24 26; do
        python run_logit_ranker.py -i music -m llama3b -l "$lrg" -a "$arg" -d gender -bs steered
    done
done
