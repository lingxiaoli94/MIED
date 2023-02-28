# Gaussians. Change quadratic_uc_2d to quadratic_uc_nd to run on dimension n.
python run.py --prob=quadratic_uc_2d --method=MIED --num_particle=500 --projector=DB --reparam=id --wandb=true

# Student's t-distributions. '--filter_range' is the box where we restrict the evaluation in (this is $a$ in Figure 2 of the paper).
python run.py --prob=student_uc_2d --method=MIED --num_particle=1000 --projector=DB --reparam=id --filter_range=5 --wandb=true

# Uniform sampling in a box with different kernels for MIED (default kernel is Riesz with `--eps=1e-8`).
python run.py --prob=uniform_box_2d --method=MIED --num_particle=500 --projector=DB --reparam=box_tanh --wandb=true
python run.py --prob=uniform_box_2d --method=MIED --kernel=laplace --eps=1e-2 --num_particle=500 --projector=DB --reparam=box_tanh --wandb=true
python run.py --prob=uniform_box_2d --method=MIED --kernel=gaussian --eps=1e-3 --num_particle=500 --projector=DB --reparam=box_tanh --wandb=true
