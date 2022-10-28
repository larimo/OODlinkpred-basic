# Sample Code - Out-of-distribution link prediction

## Introduction

This is a sample implementation of out-of-distribution link prediction in PyTorch to analyze
accuracy of various graph learning methods on out-of-distribution link prediction tasks where
the training graphs are sub-sampled from original large graphs by several sampling methods.

## Authors
[Larissa Mori](https://web.ics.purdue.edu/~lkawanom), Rajdeep Haldar

## Training Commands

#### Example:

```bash
python -u main.py --dataset sbm_1000 --node_embedding GCN \
     --eval_method Hits@50 --device cuda --hidden_channels 16 --num_layers 3 \
     --dropout 0.0 --lr 0.0001 --batch_size 128 --patience 20 --epochs 200 \
     --runs 5 --test_distribution OOD --test_dataset sbm_10000
```

## Acknowledgement
We extend the basic link prediction code from Dr. Ribeiro's lab (CS@Purdue) to the out-of-distribution case.
