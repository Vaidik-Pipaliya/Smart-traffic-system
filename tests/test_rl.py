import os
import pytest
import numpy as np
from simulation.rl_env import TrafficEnv
from simulation.dqn_agent import DQNAgent
from simulation.agent import TrafficLightAgent
from simulation.models import Intersection


def test_traffic_env_reset_and_step():
    """Verify TrafficEnv resets cleanly and returns expected state shapes and rewards."""
    env = TrafficEnv(target_intersection_id="I_1_1", episode_length=50, spawn_rate=0.4, seed=42)
    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)
    assert state[0] >= 0.0  # Queue NS non-negative
    assert state[1] >= 0.0  # Queue EW non-negative

    next_state, reward, done, info = env.step(action=1)

    assert next_state.shape == (4,)
    assert isinstance(reward, float)
    assert reward <= 0.0  # Penalty reward is non-positive
    assert isinstance(done, bool)
    assert "avg_wait_time" in info


def test_dqn_agent_act_and_replay(tmp_path):
    """Verify DQNAgent act selection, memory replay, and model weight saving/loading."""
    agent = DQNAgent(state_dim=4, action_dim=2, epsilon=0.0)  # Pure exploitation for test
    state = np.array([2.0, 1.0, 0.0, 0.5], dtype=np.float32)

    action = agent.act(state, eval_mode=True)
    assert action in (0, 1)

    # Store experience transitions
    for i in range(40):
        s = np.array([float(i), 1.0, 0.0, 0.2], dtype=np.float32)
        ns = np.array([float(i + 1), 0.0, 1.0, 0.4], dtype=np.float32)
        agent.remember(s, 1, -5.0, ns, False)

    loss = agent.replay(batch_size=32)
    assert loss is not None
    assert isinstance(loss, float)

    # Save and load model checkpoint
    model_file = tmp_path / "test_model.pth"
    agent.save(str(model_file))
    assert model_file.exists()

    agent2 = DQNAgent(state_dim=4, action_dim=2)
    agent2.load(str(model_file))
    action2 = agent2.act(state, eval_mode=True)
    assert action2 == action


def test_agent_mode_toggling():
    """Verify TrafficLightAgent toggles between FIXED and RL mode."""
    agent = TrafficLightAgent("agent_1", "I1", mode="FIXED")
    assert agent.mode == "FIXED"

    agent.set_mode("RL")
    assert agent.mode == "RL"

    with pytest.raises(ValueError, match="Invalid agent mode"):
        agent.set_mode("SUPER_INTELLIGENT")
