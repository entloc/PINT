# PN-INT: A Path and Neighbor-based Interaction Model for Few-shot Knowledge Graph Reasoning

## Start

### Requirements
* ``Python 3.7.9 ``
* ``PyTorch 1.7.0``

### Datasets
We conduct our experiments on two well-adopted benchmark datasets — NELL-One and Wiki-One. 
You can find original datasets them from [here](https://github.com/xwhan/One-shot-Relational-Learning).

### Training
* For NELL-One: python train.py --dataset "NELL-One" --embed_dim 100 --hidden_dim 100 --few 1 --eval_every 50
* For Wiki-One: python train.py --dataset "Wiki-One" --embed_dim 50 --hidden_dim 50 --few 1 --eval_every 200


