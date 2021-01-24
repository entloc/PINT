# PINT: A Path-based Interaction Model for Few-Shot Knowledge Graph Reasoning

## Start

### Requirements
* ``Python 3.7.9 ``
* ``PyTorch 1.7.0``

### Datasets
We conduct our experiments on two well-adopted benchmark datasets — NELL-One and Wiki-One. 
You can find original datasets them from [here](https://github.com/xwhan/One-shot-Relational-Learning).

You can download datasets used in this work from [here](https://pan.baidu.com/s/1ENTGLHQLU9W6m4Eb1XOx1A), the extraction code is 36yy.

### Training
* For NELL-One: python train.py --dataset "NELL-One" --embed_dim 100 --hidden_dim 100 --few 1 --eval_every 50 ----max_batches 300
* For Wiki-One: python train.py --dataset "Wiki-One" --embed_dim 50 --hidden_dim 50 --few 1 --eval_every 200 --max_batches 100

### Test
* For NELL-One: python train.py --test --dataset "NELL-One" --embed_dim 100 --hidden_dim 100 --few 1
* For Wiki-One: python train.py --test --dataset "Wiki-One" --embed_dim 50 --hidden_dim 50 --few 1

