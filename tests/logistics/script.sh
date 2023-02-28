# Bayesian logistic regression. Change `waveform` to other dataset names to try on other datasets.
# Available dataset names are defined in `run.py`.
python run.py --data_name=waveform --method=MIED --num_particle=1000 --projector=DB --reparam=id --num_itr=10000 --val_freq=1000 --traj_freq=1 --wandb=true

