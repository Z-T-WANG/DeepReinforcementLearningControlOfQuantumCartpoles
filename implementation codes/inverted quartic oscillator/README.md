
python3 main_parallel.py --input xp --init_lr 2.e-3

python3 main_parallel.py --input wavefunction --init_lr 1.e-3 --input_scaling 10.

python3 main_parallel.py --control_strategy damping --con_parameter 1.9

python3 main_parallel.py --control_strategy LQG --con_parameter 260

python3 main_parallel.py --control_strategy semiclassical


