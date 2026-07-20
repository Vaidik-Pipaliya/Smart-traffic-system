import pytest
from simulation.agent import TrafficLightAgent, VALID_STATES
from simulation.models import Intersection, Road, Vehicle


def test_agent_fixed_timer_toggles_state():
    """Verify light state toggles after phase_duration ticks."""
    agent = TrafficLightAgent("agent_1", "I1", initial_state="NORTH_SOUTH", phase_duration=5)
    assert agent.get_state() == "NORTH_SOUTH"

    # Step 4 times -> state should remain NORTH_SOUTH
    for _ in range(4):
        toggled = agent.step()
        assert not toggled
        assert agent.get_state() == "NORTH_SOUTH"

    # Step 5th time -> state toggles to EAST_WEST
    toggled = agent.step()
    assert toggled
    assert agent.get_state() == "EAST_WEST"
    assert agent.timer == 0


def test_safety_constraint_mutual_exclusion():
    """Verify opposing movement directions (NS and EW) are mutually exclusive."""
    intersection = Intersection("I1", phase_duration=10)

    # In NORTH_SOUTH phase
    assert intersection.is_green("NS") is True
    assert intersection.is_green("EW") is False
    # Verify opposing directions cannot be green simultaneously
    assert not (intersection.is_green("NS") and intersection.is_green("EW"))

    # Switch to EAST_WEST phase
    intersection.toggle_phase()
    assert intersection.is_green("NS") is False
    assert intersection.is_green("EW") is True
    # Verify opposing directions cannot be green simultaneously
    assert not (intersection.is_green("NS") and intersection.is_green("EW"))


def test_invalid_state_validation():
    """Verify agent rejects invalid states with ValueError."""
    agent = TrafficLightAgent("agent_1", "I1")

    with pytest.raises(ValueError, match="Invalid light state"):
        agent.change_state("YELLOW_FLASHING")

    with pytest.raises(ValueError, match="Invalid light state"):
        agent.change_state("ALL_GREEN")


def test_agent_observation_queue_tracking():
    """Verify agent.observe() returns accurate queue lengths and metrics."""
    intersection = Intersection("I_OBS", phase_duration=10)
    intersection.add_to_queue("R1", "veh_1")
    intersection.add_to_queue("R1", "veh_2")
    intersection.add_to_queue("R2", "veh_3")

    obs = intersection.observe()

    assert obs["agent_id"] == "agent_I_OBS"
    assert obs["intersection_id"] == "I_OBS"
    assert obs["state"] == "NORTH_SOUTH"
    assert obs["queue_lengths"]["R1"] == 2
    assert obs["queue_lengths"]["R2"] == 1
    assert obs["total_queue"] == 3
