# Fairness Bayesian neural networks. `--thres` is the threshold used in forming the constraints (this is $t$ in the paper).
python run.py --thres=0.001 --method=MIED --num_particle=50 --projector=DB --lr=0.001 --reparam=id --num_itr=2000 --val_freq=10 --save_freq=-1 --wandb=true

