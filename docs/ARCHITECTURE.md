# Living Architecture & System Design

## Overview
The Smart Multi-Agent Traffic Management System models urban traffic networks through distributed intelligent agents. Intersection traffic signals and autonomous vehicles operate as agents coordinating to reduce congestion, minimize wait times, and improve overall throughput.

## Directory Structure

| Folder | Purpose | Key Components |
|---|---|---|
| `/backend` | REST API layer & service integration | FastAPI (`main.py`), Pydantic models, health checks |
| `/frontend` | Real-time web dashboard UI | React + TypeScript + Vite, interactive grid view |
| `/simulation` | Multi-agent simulation engine | Graph network, Vehicle models, `TrafficLightAgent`, `TrafficEnv`, `DQNAgent` |
| `/docs` | Architectural & technical documentation | Architecture spec, API schemas, design records |
| `/` | Training & Root scripts | `train.py`, `model.pth`, `README.md` |

## Reinforcement Learning (DQN) Architecture

### 1. Environment Wrapper (`simulation/rl_env.py`)
The `TrafficEnv` class wraps the simulation into a Gym-like environment.

#### State Space ($\mathcal{S}$)
A 4-dimensional normalized vector:
$$s_t = \begin{bmatrix} \text{Queue}_{\text{NS}}, & \text{Queue}_{\text{EW}}, & \text{PhaseIdx}, & \text{TimerNorm} \end{bmatrix}$$
- $\text{Queue}_{\text{NS}}$: Count of vehicles queued on North-South incoming roads (validated $\ge 0$).
- $\text{Queue}_{\text{EW}}$: Count of vehicles queued on East-West incoming roads (validated $\ge 0$).
- $\text{PhaseIdx}$: $0.0$ for `"NORTH_SOUTH"` signal phase, $1.0$ for `"EAST_WEST"` signal phase.
- $\text{TimerNorm}$: Current phase timer divided by `phase_duration`.

#### Action Space ($\mathcal{A}$)
Discrete binary action space $\mathcal{A} \in \{0, 1\}$:
- `0`: **Hold / Stay** in current signal phase.
- `1`: **Switch** to the opposite signal phase.

#### Reward Function ($\mathcal{R}$)
Designed to penalize both cumulative waiting time and queue bottlenecks:
$$r_t = -\Big(\text{TotalWaitTime}_t + 2.0 \times \text{TotalQueue}_t\Big)$$

---

### 2. Deep Q-Network Agent (`simulation/dqn_agent.py`)
- **Q-Function Architecture**: 3-layer Multi-Layer Perceptron (MLP) in PyTorch (`state_dim=4` $\rightarrow 64 \rightarrow 64 \rightarrow \text{action_dim}=2$) with ReLU activation.
- **Experience Replay**: Memory buffer (`deque`, max capacity 2000) sampling random minibatches of size 32.
- **Exploration Strategy**: $\epsilon$-greedy action selection decaying from $\epsilon_0 = 1.0$ down to $\epsilon_{\text{min}} = 0.01$ at a rate of $\gamma_{\text{decay}} = 0.99$.
- **Target Network**: Periodically synchronized target network (`update_target_network()`) to stabilize Q-value targets.
- **Model Checkpoint**: Weights saved to `model.pth` via `torch.save()`.

---

### 3. Agent Integration & Modes (`simulation/agent.py`)
`TrafficLightAgent` supports dual operation modes:
- **`FIXED`**: Baseline fixed-duration timer phase switching.
- **`RL`**: Dynamic action selection driven by the trained `DQNAgent` neural network.

---

## Pending Requirements

- [x] **Phase 1: Architecture & Environment Setup**: Root configuration, FastAPI health check, Vite dashboard.
- [x] **Phase 2: Base Simulation Engine**: Graph road network, OO vehicle/intersection models, step lifecycle, CLI driver.
- [x] **Phase 3: Multi-Agent Logic**: `TrafficLightAgent` class, baseline fixed-timer logic, queue observation space, safety constraints.
- [x] **Phase 4: Deep Q-Network (DQN) Reinforcement Learning**: `TrafficEnv` wrapper, PyTorch `DQNAgent`, experience replay, `train.py` pipeline, `model.pth`.
- [ ] **Phase 5: API & Websocket Integration**: Streaming simulation metrics to frontend via FastAPI websockets.
- [ ] **Phase 6: Real-Time Visualization**: Interactive canvas / SVG traffic map grid in React dashboard.
