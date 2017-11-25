#!/bin/bash

jupyter nbconvert --to python deep-learning-ex1.ipynb
rm -rf output.log

ep=600
mb=32
hl=512
lr=( 0.1 0.01 0.001 )
kp=( 0.75 0.5 )
rt=( L2 L1 )
rn=( 1 0.1 0.01 )
mn=( 3 4 )

for _lr in ${lr[@]}
do
	for _kp in ${kp[@]}
	do
		for _rt in ${rt[@]}
		do
			for _rn in ${rn[@]}
            do
				for _mn in ${mn[@]}
            	do
					python deep-learning-ex1.py --ep=$ep --mb=$mb --hl=%hl --lr=$_lr --kp=$_kp --rt=$_rt --rn=$_rn --mn=$_mn
            	done
            done
		done
	done
done
