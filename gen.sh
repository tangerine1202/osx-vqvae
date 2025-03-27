#!/bin/bash

shape_name=vqvae
img_size=32
ep=10000
exp_name=${shape_name}_${img_size}_ep${ep}

ckpt_path="results/${exp_name}/ckpt.pth"

python3 main.py -e $ep -sn $shape_name -img-size $img_size -o $exp_name
python3 eval.py     $ckpt_path -sn $shape_name -img-size $img_size
python3 save_emb.py $ckpt_path -sn $shape_name -img-size $img_size