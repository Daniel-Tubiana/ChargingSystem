# EV Charging Management System

*This repository documents my final research project as part of my M.Sc. in Systems Engineering.*

A reinforcement learning–based system for optimizing electric vehicle charging in smart buildings with renewable energy sources.

## Objective

This project develops and evaluates reinforcement learning algorithms for managing electric vehicle (EV) charging in buildings with Electric Grid and photovoltaic (PV) systems. The goal is to optimize charging schedules while considering:

- **Power constraints**: Utilizing surplus solar power and building transformer limits (630 kVA)
- **Time constraints**: Charging vehicles within their available time windows
- **SOC requirements**: Ensuring vehicles reach required State of Charge (SOC) levels for planned trips
- **Energy efficiency**: Maximizing renewable energy utilization while minimizing grid dependency

A single RL agent controls actions for up to **189 EVs** (MultiDiscrete action space) in a building with **126 apartments**; algorithms evaluated include **A2C** and **PPO**.

### The RL process in the developed environment
<img width="997" height="592" alt="image" src="https://github.com/user-attachments/assets/6177ef09-eb32-4b8f-9c41-a9e6f91aefb0" />

## Data Simulation

The system includes comprehensive data simulation modules that generate realistic datasets for training and evaluation:

### Simulated Data Components
- **User Data**: EV user profiles including arrival times, time‑until‑departure, and energy requirements
- **EV Characteristics**: Battery capacity, charging rates, and initial SOC for 8 popular EV models in Israel
- **Building Power Data**:
  - Solar PV power generation based on real irradiation data
  - Household electricity consumption patterns
  - Grid connection limits and transformer capacity
- **Electricity Pricing**: Time‑of‑use pricing for grid electricity and fixed PV pricing tiers

### Data Files Generated
- `usr_ev_data_train.xlsx`: Training dataset with EV/user data
- `usr_ev_data_test.xlsx`: Test dataset for model evaluation
- `Building_data.xlsx`: Building power profiles and surplus power availability
- Additional pricing and configuration files

### Simulation Parameters
- **Building capacity**: 126 apartments/houses, 630 kVA transformer
- **EV adoption rates**: 100% for development (189 EVs), 78% for evaluation (148 EVs)
- **Charging infrastructure**: 0 kW, 3.7 kW, and 11 kW charging actions
- **Time horizon**: 24‑hour optimization periods

## Environments

Two reinforcement learning environments with different **reward structures**:

### Environment Types
1. **Continuous‑reward environment** (`EVChargingEnv_continuous`)
   - Smooth, continuous reward shaping
   - Exponential penalties for constraint violations
   - Gradual SOC‑based rewards encouraging optimal charging

2. **Discrete‑reward environment** (`EVChargingEnv_discrete`)
   - Binary/step rewards
   - Step‑function penalties for violations
   - Clear success/failure signal for SOC requirements

### Environment Features
- **MultiDiscrete action space**: 3 charging levels per EV (0 kW, 3.7 kW, 11 kW)
- **Rich observation space**: EV states, available surplus power, and time
- **Vectorized environments**: Parallel training (e.g., 16 environments)
- **Gym/Gymnasium‑compatible**: Standard RL interface

### Reward Components
- **Time‑window rewards**: Penalties for charging outside an EV’s availability
- **Power‑limit rewards**: Penalties for exceeding surplus power
- **SOC rewards**: Incentives for reaching required charge levels

## Training & Evaluation Results

### Algorithm Performance (per the project report)
**A2C (Advantage Actor‑Critic)**
- Training time: ~**1.38 h** (100% adoption), ~**1.07 h** (78% adoption)
- Better performance and faster convergence
- More stable training; respected power limits more consistently

**PPO (Proximal Policy Optimization)**
- Training time: ~**4.76 h** (100% adoption), ~**3.76 h** (78% adoption)
- More oscillatory training behavior and frequent limit violations
- Underperformed A2C across the measured metrics

### Key Metrics
- **Total reward** and **reward variance**
- **Power utilization** and **charge‑power variance**
- **SOC achievement** (vehicles meeting minimum SOC + trip energy)
- **Constraint violations** (power/time window breaches)
- **Mean SOC deviation** for vehicles below target

### Training Configuration
- **Total timesteps**: up to ~1,358,000
- **Evaluation/Logging**: periodic evaluation; TensorBoard logs
- **Normalization**: observation/reward normalization (as used in code)

## Analyze Results

Tools include:
- **Charge Profile Analysis** (24‑hour charge power vs. surplus)
- **SOC Analysis** (distributions and attainment rates)
- **Power Usage Analysis** (surplus utilization and violations)
- **Comparative Analysis** (across reward structures and algorithms)

Outputs:
- Excel result files with multiple analysis sheets
- Automated plots/figures and statistical summaries


## Project Structure

```
ChargingSystem/
├── config/                  # Configuration files
│   ├── simulation_params.yaml  # Parameters for simulation
│   └── train_config.yaml       # Training configuration
├── legacy/                  # Original implementations
├── src/                     # Source code
│   ├── data/                # Data handling modules
│   │   ├── raw/             # Raw data
│   │   ├── simulated/       # Simulated data
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── data_simulation.ipynb  # Data simulation notebook
│   ├── environments/        # Reinforcement Learning environments
│   │   ├── __init__.py
│   │   ├── base_env.py                    # Base environment class
│   │   ├── environment_factory.py         # Factory for environment creation
│   │   ├── ev_charging_env.py             # EV charging environment
│   │   ├── ev_charging_env_continuous.py  # Continuous action EV environment
│   │   └── ev_charging_env_discrete.py    # Discrete action EV environment
│   ├── evaluation/          # Evaluation scripts
│   │   ├── __init__.py
│   │   └── evaluator.py     # Model evaluation utilities
│   ├── logs/                # Log files storage
│   ├── mlruns/              # MLFlow runs for experiments
│   ├── models/              # Model definitions and scripts
│   │   ├── __init__.py
│   │   └── rl_models.py     # Reinforcement Learning models
│   ├── outputs/             # Output results from runs
│   ├── results/             # Results and metrics
│   └── utils/               # Utility functions
│   ├── __init__.py
│   └── main.py          # Main training loop
├── .gitattributes           
├── .gitignore               
└── README.md                # Project documentation
```

## Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Generate data**: Run data simulation to create training datasets
3. **Configure**: Edit `config/train_config.yaml` for your setup
4. **Train**: `python src/main.py` to start training
5. **Evaluate**: Models are automatically evaluated during training
6. **Analyze**: Use analysis tools to examine results

## Requirements

- Python 3.8+
- PyTorch (via Stable-Baselines3)
- Gymnasium
- Pandas, NumPy
- Hydra for configuration management
- Additional dependencies in requirements.txt


## Data Availability

Some source datasets and generated artifacts are large and were not uploaded to this repository.

**Not included (examples):**
- household-load dataset used to derive consumption profiles
- Trained model checkpoints and experiment logs (e.g., `models/`, `mlruns/`, and TensorBoard logs)

**How to reproduce locally:**
- Run the data simulation notebook/script (`src/data/data_simulation.ipynb`) to regenerate the training/test datasets and building power profiles.
- Train models with the provided configs (`config/train_config.yaml`); outputs will be written to `outputs/` and `results/`.

If you need the original large files, please contact me and I can share them (e.g., via Drive) or add them as a release asset. 