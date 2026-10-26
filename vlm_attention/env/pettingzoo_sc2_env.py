"""
PettingZoo ParallelEnv wrapper for the existing SC2MultimodalEnv.

Wraps the StarCraft II environment in a PettingZoo-compatible multi-agent interface.

Each friendly unit is represented as an agent. All active friendly unit agents
submit one discrete action at a time, and those actions are combined into the
single team-level action dictionary expected by SC2MultimodalEnv.

Action meanings:
    0 = hold
    1 = attack the nearest visible enemy
    2 = move up
    3 = move right
    4 = move down
    5 = move left
    6 = retreat away from the nearest visible enemy

This wrapper does not train an RL policy and does not make one Gemini call per
unit. It provides the multi-agent interface needed for those next steps.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from vlm_attention.env.env_core import SC2MultimodalEnv


class SC2UnitParallelEnv(ParallelEnv):
    metadata = {
        "name": "sc2_unit_parallel_v0",
        "render_modes": ["human"],
        "is_parallelizable": True,
    }

    # Features for each unit row:
    # [friendly, x_norm, y_norm, health_fraction, shield_fraction,
    #  energy_norm, is_alive, is_this_agent]
    UNIT_FEATURES = 8
    SELF_FEATURES = 8

    def __init__(
        self,
        *,
        map_name: str,
        save_dir: str,
        replay_dir: str,
        feature_dims: tuple[int, int],
        rgb_dims: tuple[int, int],
        map_size: tuple[int, int],
        max_friendly_units: int = 16,
        max_observed_units: int = 32,
        max_cycles: int = 125,
        render_mode: str | None = None,
    ):
        super().__init__()

        if max_friendly_units < 1:
            raise ValueError("max_friendly_units must be at least 1.")
        if max_observed_units < max_friendly_units:
            raise ValueError(
                "max_observed_units must be >= max_friendly_units."
            )
        if max_cycles < 1:
            raise ValueError("max_cycles must be at least 1.")

        self.render_mode = render_mode
        self.max_friendly_units = int(max_friendly_units)
        self.max_observed_units = int(max_observed_units)
        self.max_cycles = int(max_cycles)
        self.map_size = tuple(map_size)
        self.rgb_dims = tuple(rgb_dims)

        self.possible_agents = [
            f"unit_{i}" for i in range(self.max_friendly_units)
        ]
        self.agents: list[str] = []

        self._base_env_kwargs = {
            "map_name": map_name,
            "save_dir": save_dir,
            "replay_dir": replay_dir,
            "feature_dims": feature_dims,
            "rgb_dims": rgb_dims,
            "map_size": map_size,
        }
        self.env: SC2MultimodalEnv | None = None

        # Each agent receives the shared RGB frame, a compact feature vector
        # for itself, and a padded table describing all currently observed units.
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(*self.rgb_dims, 3),
                        dtype=np.uint8,
                    ),
                    "self_features": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.SELF_FEATURES,),
                        dtype=np.float32,
                    ),
                    "unit_features": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(
                            self.max_observed_units,
                            self.UNIT_FEATURES,
                        ),
                        dtype=np.float32,
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(7) for agent in self.possible_agents
        }

        self._agent_to_tag: dict[str, int] = {}
        self._tag_to_agent: dict[int, str] = {}
        self._unit_info: list[dict[str, Any]] = []
        self._cycle = 0

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    @staticmethod
    def _position(unit: dict[str, Any]) -> tuple[float, float]:
        """Prefer grid_position, falling back to position; safely parse either."""
        for key in ("grid_position", "position"):
            value = unit.get(key)
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
                try:
                    return float(value[0]), float(value[1])
                except (TypeError, ValueError):
                    pass
            if isinstance(value, dict) and "x" in value and "y" in value:
                try:
                    return float(value["x"]), float(value["y"])
                except (TypeError, ValueError):
                    pass
        return 0.0, 0.0

    @staticmethod
    def _fraction(value: Any, maximum: Any) -> float:
        try:
            maximum_value = float(maximum)
            if maximum_value <= 0:
                return 0.0
            return float(np.clip(float(value) / maximum_value, 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.0

    def _friendly_units(self) -> list[dict[str, Any]]:
        return [u for u in self._unit_info if int(u.get("alliance", -1)) == 1]

    def _enemy_units(self) -> list[dict[str, Any]]:
        return [u for u in self._unit_info if int(u.get("alliance", -1)) == 4]

    def _nearest_enemy(self, unit: dict[str, Any]):
        enemies = self._enemy_units()
        if not enemies:
            return None
        ux, uy = self._position(unit)
        return min(
            enemies,
            key=lambda enemy: (
                (self._position(enemy)[0] - ux) ** 2
                + (self._position(enemy)[1] - uy) ** 2
            ),
        )

    def _make_observation(self, agent: str, image: np.ndarray):
        tag = self._agent_to_tag[agent]
        self_unit = next(
            (
                unit
                for unit in self._friendly_units()
                if int(unit.get("simplified_tag", -1)) == tag
            ),
            None,
        )

        self_features = np.zeros(self.SELF_FEATURES, dtype=np.float32)
        if self_unit is not None:
            x, y = self._position(self_unit)
            map_w = max(float(self.map_size[0]), 1.0)
            map_h = max(float(self.map_size[1]), 1.0)
            nearest = self._nearest_enemy(self_unit)
            dx = dy = enemy_health = 0.0
            if nearest is not None:
                ex, ey = self._position(nearest)
                dx = float(np.clip((ex - x) / map_w, -1.0, 1.0))
                dy = float(np.clip((ey - y) / map_h, -1.0, 1.0))
                enemy_health = self._fraction(
                    nearest.get("health", 0), nearest.get("max_health", 0)
                )
            # Offset signed relative positions into [0, 1].
            self_features[:] = [
                self._fraction(self_unit.get("health", 0), self_unit.get("max_health", 0)),
                self._fraction(self_unit.get("shield", 0), self_unit.get("max_shield", 0)),
                float(np.clip(x / map_w, 0.0, 1.0)),
                float(np.clip(y / map_h, 0.0, 1.0)),
                (dx + 1.0) / 2.0,
                (dy + 1.0) / 2.0,
                enemy_health,
                1.0,
            ]

        unit_features = np.zeros(
            (self.max_observed_units, self.UNIT_FEATURES),
            dtype=np.float32,
        )
        # Deterministic order: friendly units first, then enemy units, by tag.
        ordered = sorted(
            self._unit_info,
            key=lambda u: (
                0 if int(u.get("alliance", -1)) == 1 else 1,
                int(u.get("simplified_tag", 0)),
            ),
        )[: self.max_observed_units]

        for index, unit in enumerate(ordered):
            x, y = self._position(unit)
            is_friendly = int(unit.get("alliance", -1)) == 1
            unit_tag = int(unit.get("simplified_tag", -1))
            unit_features[index] = [
                1.0 if is_friendly else 0.0,
                float(np.clip(x / max(float(self.map_size[0]), 1.0), 0.0, 1.0)),
                float(np.clip(y / max(float(self.map_size[1]), 1.0), 0.0, 1.0)),
                self._fraction(unit.get("health", 0), unit.get("max_health", 0)),
                self._fraction(unit.get("shield", 0), unit.get("max_shield", 0)),
                self._fraction(unit.get("energy", 0), 200.0),
                1.0,
                1.0 if is_friendly and unit_tag == tag else 0.0,
            ]

        # Use the same frame object for all agents rather than copying it.
        return {
            "image": np.asarray(image, dtype=np.uint8),
            "self_features": self_features,
            "unit_features": unit_features,
        }

    def _observations(self, observation):
        image = observation["image"]
        return {
            agent: self._make_observation(agent, image)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if self.env is not None:
            self.env.close()

        self.env = SC2MultimodalEnv(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            **self._base_env_kwargs,
        )
        observation = self.env.reset(seed=seed, options=options)
        self._unit_info = list(observation["unit_info"])
        friendly = sorted(
            self._friendly_units(),
            key=lambda unit: int(unit.get("simplified_tag", 0)),
        )

        if len(friendly) > self.max_friendly_units:
            self.env.close()
            self.env = None
            raise ValueError(
                f"Found {len(friendly)} friendly units, but "
                f"max_friendly_units={self.max_friendly_units}. Increase it."
            )

        self._agent_to_tag = {
            f"unit_{index}": int(unit["simplified_tag"])
            for index, unit in enumerate(friendly)
        }
        self._tag_to_agent = {
            tag: agent for agent, tag in self._agent_to_tag.items()
        }
        self.agents = list(self._agent_to_tag)
        self._cycle = 0

        infos = {
            agent: {
                "unit_tag": self._agent_to_tag[agent],
                "unit_name": next(
                    (
                        u.get("unit_name", "unknown")
                        for u in friendly
                        if int(u["simplified_tag"]) == self._agent_to_tag[agent]
                    ),
                    "unknown",
                ),
            }
            for agent in self.agents
        }
        return self._observations(observation), infos

    def step(self, actions):
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
        if not self.agents:
            return {}, {}, {}, {}, {}

        unexpected = set(actions) - set(self.agents)
        if unexpected:
            raise ValueError(f"Actions provided for inactive agents: {unexpected}")

        attacks = []
        moves = []
        alive_friendly = {
            int(u.get("simplified_tag", -1)): u
            for u in self._friendly_units()
        }

        # Actions are simultaneous. Missing actions default to HOLD.
        for agent in list(self.agents):
            unit_tag = self._agent_to_tag[agent]
            unit = alive_friendly.get(unit_tag)
            if unit is None:
                continue
            action = int(actions.get(agent, 0))
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action {action} for {agent}")

            if action == 0:
                continue
            elif action == 1:
                target = self._nearest_enemy(unit)
                if target is not None:
                    attacks.append(
                        (unit_tag, int(target["simplified_tag"]))
                    )
            elif action in (2, 3, 4, 5):
                # SC2MultimodalEnv uses move_type=2 for directional movement:
                # 0=up, 1=right, 2=down, 3=left.
                direction = action - 2
                moves.append((2, unit_tag, [direction, 0]))
            elif action == 6:
                target = self._nearest_enemy(unit)
                if target is not None:
                    ux, uy = self._position(unit)
                    ex, ey = self._position(target)
                    # Choose the dominant axis pointing away from the enemy.
                    if abs(ux - ex) >= abs(uy - ey):
                        direction = 1 if ux >= ex else 3  # right or left
                    else:
                        direction = 2 if uy >= ey else 0  # down or up
                    moves.append((2, unit_tag, [direction, 0]))

        next_observation, reward, done, info = self.env.step(
            {"attack": attacks, "move": moves}
        )
        self._cycle += 1

        if next_observation is not None:
            self._unit_info = list(next_observation["unit_info"])

        # Units that disappeared from the observation are removed from the
        # active agent list; the tag-to-agent mapping never gets reassigned.
        surviving_tags = {
            int(u.get("simplified_tag", -1))
            for u in self._friendly_units()
        }
        previous_agents = list(self.agents)
        dead_agents = [
            agent
            for agent in previous_agents
            if self._agent_to_tag[agent] not in surviving_tags
        ]
        self.agents = [
            agent for agent in previous_agents if agent not in dead_agents
        ]

        terminated = bool(done)
        truncated = self._cycle >= self.max_cycles and not terminated

        if terminated or truncated:
            ended_agents = previous_agents
            observations = {
                agent: self._make_observation(
                    agent,
                    next_observation["image"]
                    if next_observation is not None
                    else np.zeros(
                        (*self.rgb_dims, 3),
                        dtype=np.uint8,
                    ),
                )
                for agent in ended_agents
            }
            rewards = {agent: float(reward) for agent in ended_agents}
            terminations = {agent: terminated for agent in ended_agents}
            truncations = {agent: truncated for agent in ended_agents}
            infos = {agent: dict(info or {}) for agent in ended_agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        if next_observation is None:
            # Underlying environment reported an error and ended the episode.
            observations = {}
            rewards = {agent: 0.0 for agent in previous_agents}
            terminations = {agent: True for agent in previous_agents}
            truncations = {agent: False for agent in previous_agents}
            infos = {agent: dict(info or {}) for agent in previous_agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        observations = self._observations(next_observation)
        rewards = {agent: float(reward) for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: dict(info or {}) for agent in self.agents}
        for agent in dead_agents:
            rewards[agent] = float(reward)
            terminations[agent] = True
            truncations[agent] = False
            infos[agent] = {"unit_eliminated": True}

        return observations, rewards, terminations, truncations, infos

    def state(self):
        """Return a centralized state vector for centralized-training methods."""
        if self._unit_info is None:
            return np.zeros(
                (self.max_observed_units, self.UNIT_FEATURES),
                dtype=np.float32,
            )
        state = np.zeros(
            (self.max_observed_units, self.UNIT_FEATURES),
            dtype=np.float32,
        )
        ordered = sorted(
            self._unit_info,
            key=lambda u: (
                0 if int(u.get("alliance", -1)) == 1 else 1,
                int(u.get("simplified_tag", 0)),
            ),
        )[: self.max_observed_units]
        for index, unit in enumerate(ordered):
            x, y = self._position(unit)
            friendly = int(unit.get("alliance", -1)) == 1
            state[index] = [
                1.0 if friendly else 0.0,
                float(np.clip(x / max(float(self.map_size[0]), 1.0), 0.0, 1.0)),
                float(np.clip(y / max(float(self.map_size[1]), 1.0), 0.0, 1.0)),
                self._fraction(unit.get("health", 0), unit.get("max_health", 0)),
                self._fraction(unit.get("shield", 0), unit.get("max_shield", 0)),
                self._fraction(unit.get("energy", 0), 200.0),
                1.0,
                0.0,
            ]
        return state

    def render(self):
        # The underlying SC2 environment already opens its visualizer.
        return None

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.agents = []
