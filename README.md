# flock: Multi-Agent Reinforcement Learning for Cooperative Transport

A reinforcement learning environment modeled after OpenAI Gym, designed for studying multi-agent cooperation.

## Installation

```bash
# Install from PyPI
pip install flock-rl

# Or install from source
git clone https://github.com/kzqiu/flock.git
cd flock
pip install -e .
```

## Training
Use the files `examples/maddpg.py` to train MATD3 or `examples/stable_baselines_td3.ipynb` to train TD3. There are no arguments for running these files so you must go into the file in order to modify the training hyperparameters. 

The models will be saved into the `models/` directory and their evaluated results will be saved as `.npy/.npz` files which can be read into numpy arrays using `np.load`.

## Running Simulations
To run a simple, deterministic controller, run:

```bash
python flock/simulation.py
```

To run a trained model, load the desired policy for MATD3 in `examples/maddpg_eval.py` or for TD3 `examples/stable_baselines_td3.ipynb`.

## Environment
Please see the paper for additional details about the environment design. To modify how the environment is used, see `flock/environment/flock_env.py`.

## Project Overview
```raw
.
├── flock/
│   ├── environment/
│   │   ├── flock_env.py
│   │   ├── agent.py
│   │   ├── obstacle.py
│   │   └── transport_object.py
│   ├── assets/
│   │   └── ...
│   ├── deterministic_controller.py
│   ├── simulation.py
│   └── maddpg.py
├── examples/
│   ├── maddpg.py (MADDPG/MATD3 training)
│   ├── maddpg_eval.py
│   ├── stable_baselines_td3.ipynb (TD3 training)
│   ├── analysis.ipynb
│   └── ...
├── models/
│   ├── eval_reward_{n}.npy (training eval. for MADDPG/MATD3)
│   ├── eval_reward_{n}_td3.npy (training eval. for TD3)
│   ├── flock_{n}agent_actor.pth (actor weights for MADDPG/MATD3)
│   └── flock_td3_{n}.zip (model weights for TD3)
├── LICENSE  
├── pyproject.toml
├── README.md
├── setup.cfg
└── setup.py
```
