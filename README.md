# EV Charging Management System

A reinforcement learning-based system for optimizing electric vehicle charging in smart buildings with renewable energy sources.

## Objective

This project develops and evaluates reinforcement learning algorithms for managing electric vehicle (EV) charging in buildings with Electric Grid and photovoltaic (PV) systems. The goal is to optimize charging schedules while considering:

- **Power constraints**: Utilizing surplus solar power and building transformer limits (630 KVA)
- **Time constraints**: Charging vehicles within their available time windows
- **SOC requirements**: Ensuring vehicles reach required State of Charge (SOC) levels for planned trips
- **Energy efficiency**: Maximizing renewable energy utilization while minimizing grid dependency
- 

<img width="997" height="592" alt="image" src="https://github.com/user-attachments/assets/6177ef09-eb32-4b8f-9c41-a9e6f91aefb0" />


The system supports multi-agent environments with up to 189 EVs in a building with 126 apartments, using A2C and PPO algorithms for decision-making.

## Data Simulation

The system includes comprehensive data simulation modules that generate realistic datasets for training and evaluation:

### Simulated Data Components
- **User Data**: EV user profiles including arrival times, departure times, and energy requirements
- **EV Characteristics**: Battery capacity, charging rates, and initial SOC for 8 popular EV models in Israel
- **Building Power Data**: 
  - Solar PV power generation based on real irradiation data
  - Household electricity consumption patterns
  - Grid connection limits and transformer capacity
- **Electricity Pricing**: Time-of-use pricing for grid electricity and fixed PV pricing

### Data Files Generated
- `usr_ev_data_train.xlsx`: Training dataset with EV user data
- `usr_ev_data_test.xlsx`: Test dataset for model evaluation  
- `Building_data.xlsx`: Building power profiles and surplus power availability
- Additional pricing and configuration files

### Simulation Parameters
- **Building capacity**: 126 apartments/houses, 630 KVA transformer
- **EV adoption rates**: 100% for development (189 EVs), 78% for evaluation (148 EVs)
- **Charging infrastructure**: 3.7kW and 11kW charging options
- **Time horizon**: 24-hour optimization periods

## Environments

The system provides two reinforcement learning environments with different reward structures:

### Environment Types
1. **Continuous Reward Environment** (`EVChargingEnv_continues_reward`)
   - Smooth, continuous reward functions
   - Exponential penalty functions for constraint violations
   - Gradual SOC-based rewards encouraging optimal charging

2. **Discrete Reward Environment** (`EVChargingEnv_discrete_reward`)  
   - Binary reward structure
   - Step-function penalties for violations
   - Clear success/failure rewards for SOC requirements

### Environment Features
- **Multi-discrete action space**: 3 charging levels per EV (0kW, 3.7kW, 11kW)
- **Complex observation space**: EV states, power availability, and time information
- **Vectorized environments**: Support for parallel training with 16 environments
- **Gymnasium compliance**: Standard RL interface for algorithm compatibility

### Reward Components
- **Time constraint rewards**: Penalties for charging outside available windows
- **Power constraint rewards**: Penalties for exceeding available surplus power
- **SOC rewards**: Incentives for reaching required charge levels

## Training & Evaluation Results

### Algorithm Performance
The system has been tested with two main RL algorithms:

**A2C (Advantage Actor-Critic)**
- Training time: ~1.4 hours (100% adoption), ~1.1 hours (78% adoption)
- Better performance and faster convergence
- More stable training process

**PPO (Proximal Policy Optimization)**  
- Training time: ~4.8 hours (100% adoption), ~3.8 hours (78% adoption)
- Slower convergence but good final performance
- More oscillatory training behavior

### Key Metrics
- **Total reward tracking**: Cumulative reward across all EVs and time steps
- **Power utilization**: Efficiency of surplus power usage
- **SOC achievement**: Percentage of EVs reaching required charge levels
- **Constraint violations**: Frequency of power and time constraint breaches

### Training Configuration
- **Total timesteps**: Up to 1,358,000 steps for comprehensive training
- **Evaluation frequency**: Regular checkpoints with evaluation callback
- **Normalization**: Observation and reward normalization for stable training
- **Logging**: TensorBoard integration for training monitoring

## Analyze Results

The system includes comprehensive analysis tools for evaluating model performance:

### Analysis Components
- **Charge Profile Analysis**: Visualization of charging patterns over 24-hour periods
- **SOC Analysis**: State of charge evolution and final achievement rates
- **Power Usage Analysis**: Surplus power utilization and constraint adherence
- **Comparative Analysis**: Performance comparison across different reward structures

### Key Performance Indicators
- **Mean total reward**: Overall system performance metric
- **Power utilization efficiency**: Percentage of available surplus power used
- **SOC success rate**: Percentage of vehicles achieving required charge levels
- **Constraint violation frequency**: Count of power and time violations
- **Charging variance**: Consistency of charging decisions across episodes

### Visualization Tools
- Charge power profiles over time
- SOC distribution analysis  
- Power reward vs. power delta relationships
- Statistical summaries and performance tables

### Results Export
- Excel-based result files with multiple analysis sheets
- Automated report generation from result directories
- Statistical analysis and comparison tools
- Performance visualization and figure generation

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
│       ├── __init__.py
│       └── main.py          # Main training loop
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
