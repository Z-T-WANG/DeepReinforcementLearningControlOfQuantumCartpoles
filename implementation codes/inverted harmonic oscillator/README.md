When using measurement outcomes as the input, the network is deeper and harder to train, and the reinforcement learning process is not as stable as the other cases, and especially, its performance first increases and then decreases during learning. We only consider the results at the end of the learning progress as valid. 

The commands that we have used to produce our results are\
```python3 main_parallel.py --input xp```\
```python3 main_parallel.py --input wavefunction --input_scaling 10 --train_episodes_multiplicative 1.4```\
```python3 main_parallel.py --input measurements --input_scaling 0.1 --train_episodes_multiplicative 2. --init_lr 4e-5```

and to test the trained models, simply add argument ```--test```.

The LQG control is estimated by ```python3 main_parallel.py --LQG```

