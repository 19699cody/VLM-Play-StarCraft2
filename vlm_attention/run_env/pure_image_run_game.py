from fileinput import filename
import time
from datetime import datetime
from absl import app
from absl import flags
import os
import cv2

from vlm_attention.env.env_core import SC2MultimodalEnv
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.vlm.gemini_vlm import (
    GeminiVLM,
    AGGRESSIVE_PERSONALITY,
    PASSIVE_PERSONALITY
)


"""
Run the environment with a Gemini VLM making decisions.

Runs the game agent and connects model decisions to in-game actions.

Pipeline:

SC2 Environment
    -> screenshot + unit information
    -> Gemini VLM
    -> constrained command
    -> SC2 action
    -> repeat
"""


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "map",
    "vlm_attention_2vAI",
    "Name of the map to use"
)

flags.DEFINE_string(
    "replay_dir",
    "./replays",
    "Directory to save replays"
)

flags.DEFINE_string(
    "config_path",
    os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH),
    "Path to the configuration file"
)


class PureImageAgent:

    def __init__(self, action_space):

        self.action_space = action_space

        # Keep screenshot saving for the assignment.
        self.screenshot_dir = "./pure_screenshots"
        self.screenshot_count = 0

        # Ask the VLM every 10 environment steps.
        self.observation_interval = 10
        self.step_count = 0

        # Gemini VLM
        # Player 1: Aggressive Gemini commander
        self.gemini_player_1 = GeminiVLM(
        personality=AGGRESSIVE_PERSONALITY
        )

        # Player 2: Passive Gemini commander
        self.gemini_player_2 = GeminiVLM(
        personality=PASSIVE_PERSONALITY
        )

        os.makedirs(self.screenshot_dir, exist_ok=True)

    def get_action(self, observation):

        self.step_count += 1

        # Ask Gemini periodically.
        if self.step_count % self.observation_interval == 0:

            screenshot_path = self.capture_observation(
                observation
            )

            decision = self.gemini_player_1.decide(
                screenshot_path,
                observation["unit_info"]
            )

            print("\n" + "=" * 60)
            print("VLM DECISION")
            print("=" * 60)
            print(f"Command: {decision['command']}")
            print(f"Reason:  {decision['reason']}")
            print("=" * 60)

            action = self.command_to_action(
                decision["command"],
                observation["unit_info"]
            )

            print("SC2 ACTION")
            print("=" * 60)
            print(action)
            print("=" * 60)

            return action

        # Between VLM decisions, do nothing.
        return {
            "attack": [],
            "move": []
        }

    def capture_observation(self, observation):

        self.screenshot_count += 1

        # Save the RGB screenshot with the correct color channels.
        frame = observation["image"]
        filename = os.path.join(
            self.screenshot_dir,
            f"observation_{self.screenshot_count:04d}.png"
        )

        cv2.imwrite(
            filename,
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )

        # Convert unit information into readable text.
        game_state = self.format_unit_info(
            observation["unit_info"]
        )

        print("\n" + "=" * 60)
        print(f"OBSERVATION #{self.screenshot_count}")
        print(f"GAME STEP: {self.step_count}")
        print("=" * 60)

        print(game_state)

        print("=" * 60)
        print(f"Screenshot saved: {filename}")
        print("=" * 60)

        # Return the screenshot path so Gemini can use it.
        return filename

    def format_unit_info(self, unit_info):

        friendly_units = []
        enemy_units = []

        for unit in unit_info:

            # In this environment:
            # alliance 1 = friendly
            # alliance 4 = enemy
            alliance = unit["alliance"]

            unit_description = (
                f"{unit['unit_name']} "
                f"(tag={unit['simplified_tag']})\n"
                f"  Health: {unit['health']}/{unit['max_health']}\n"
                f"  Position: {unit['position']}"
            )

            if alliance == 1:
                friendly_units.append(unit_description)

            elif alliance == 4:
                enemy_units.append(unit_description)

        result = "FRIENDLY UNITS:\n"

        if friendly_units:
            result += "\n\n".join(friendly_units)
        else:
            result += "None visible"

        result += "\n\nENEMY UNITS:\n"

        if enemy_units:
            result += "\n\n".join(enemy_units)
        else:
            result += "None visible"

        return result

    def command_to_action(self, command, unit_info):

        """
        Convert the VLM command into the action format
        expected by SC2MultimodalEnv.

        Supported commands:

            ATTACK <enemy>
            MOVE <unit> <x,y>
            RETREAT <unit>
            HOLD
        """

        command = command.strip()

        # ==================================================
        # HOLD
        # ==================================================

        if command.upper() == "HOLD":

            return {
                "attack": [],
                "move": []
            }

        # ==================================================
        # ATTACK <enemy>
        # ==================================================

        if command.upper().startswith("ATTACK "):

            target_name = command[7:].strip()

            friendly_tags = []
            enemy_tag = None

            for unit in unit_info:

                if unit["alliance"] == 1:

                    friendly_tags.append(
                        unit["simplified_tag"]
                    )

                elif unit["alliance"] == 4:

                    if (
                        unit["unit_name"].lower()
                        == target_name.lower()
                    ):
                        enemy_tag = unit["simplified_tag"]

            if enemy_tag is None:

                print(
                    f"WARNING: Could not find enemy "
                    f"'{target_name}'."
                )

                return {
                    "attack": [],
                    "move": []
                }

            # ATTACK <enemy> means all surviving
            # friendly units attack that enemy.
            attacks = [
                (friendly_tag, enemy_tag)
                for friendly_tag in friendly_tags
            ]

            return {
                "attack": attacks,
                "move": []
            }

        # ==================================================
        # RETREAT <unit>
        # ==================================================

        if command.upper().startswith("RETREAT "):

            unit_name = command[8:].strip()

            retreat_unit = None

            for unit in unit_info:

                if (
                    unit["alliance"] == 1
                    and unit["unit_name"].lower()
                    == unit_name.lower()
                ):

                    retreat_unit = unit
                    break

            if retreat_unit is None:

                print(
                    f"WARNING: Could not find friendly "
                    f"unit '{unit_name}'."
                )

                return {
                    "attack": [],
                    "move": []
                }

            # Move away from the nearest visible enemy.
            # Direction encoding:
            # 0 = up
            # 1 = right
            # 2 = down
            # 3 = left

            enemy_units = [
                unit for unit in unit_info
                if unit["alliance"] == 4
            ]

            direction = 3  # Default to left if no enemy is visible.

            if enemy_units:
                ux, uy = retreat_unit["position"]

                nearest_enemy = min(
                    enemy_units,
                    key=lambda enemy: (
                        (ux - enemy["position"][0]) ** 2
                        + (uy - enemy["position"][1]) ** 2
                    )
                )

                ex, ey = nearest_enemy["position"]

                # Move along the axis with the larger separation.
                if abs(ux - ex) >= abs(uy - ey):
                    direction = 1 if ux >= ex else 3
                else:
                    direction = 2 if uy >= ey else 0

            move = [
                (
                    2,
                    retreat_unit["simplified_tag"],
                    [direction, 0]
                )
            ]

            return {
                "attack": [],
                "move": move
            }

        # ==================================================
        # MOVE <unit> <x,y>
        # ==================================================

        if command.upper().startswith("MOVE "):

            move_text = command[5:].strip()

            # Expected format:
            #
            # MOVE <unit> <x,y>
            #
            # Example:
            #
            # MOVE Stalker_1 7,5

            parts = move_text.rsplit(" ", 1)

            if len(parts) != 2:

                print(
                    f"WARNING: Invalid MOVE command: "
                    f"{command}"
                )

                return {
                    "attack": [],
                    "move": []
                }

            unit_name = parts[0].strip()
            location_text = parts[1].strip()

            # Parse x,y.
            try:

                x_str, y_str = location_text.split(",")

                x = int(x_str)
                y = int(y_str)

            except (ValueError, TypeError):

                print(
                    f"WARNING: Invalid MOVE location: "
                    f"{location_text}"
                )

                return {
                    "attack": [],
                    "move": []
                }

            # Grid coordinates must be 1 through 10.
            if not (1 <= x <= 10 and 1 <= y <= 10):

                print(
                    f"WARNING: MOVE location must be "
                    f"between 1 and 10: ({x}, {y})"
                )

                return {
                    "attack": [],
                    "move": []
                }

            # Find the friendly unit.
            unit_tag = None

            for unit in unit_info:

                if (
                    unit["alliance"] == 1
                    and unit["unit_name"].lower()
                    == unit_name.lower()
                ):

                    unit_tag = unit["simplified_tag"]
                    break

            if unit_tag is None:

                print(
                    f"WARNING: Could not find friendly "
                    f"unit '{unit_name}'."
                )

                return {
                    "attack": [],
                    "move": []
                }

            # Environment grid movement format:
            #
            # (move_type, unit_tag, [x, y])
            #
            # move_type = 1 means grid-based movement.

            move = [
                (
                    1,
                    unit_tag,
                    [x, y]
                )
            ]

            return {
                "attack": [],
                "move": move
            }

        # ==================================================
        # UNKNOWN COMMAND
        # ==================================================

        print(
            f"WARNING: Unknown VLM command: {command}"
        )

        return {
            "attack": [],
            "move": []
        }


def run_episode(env, agent, max_steps=100):

    observation = env.reset()

    total_reward = 0
    done = False
    step = 0

    # Give StarCraft II a few seconds to initialize.
    time.sleep(3)

    while not done and step < max_steps:

        action = agent.get_action(observation)

        observation, reward, done, _ = env.step(action)

        total_reward += reward
        step += 1

        print(
            f"Step: {step}, "
            f"Reward: {reward}, "
            f"Total Reward: {total_reward}"
        )

    return total_reward


def main(argv):

    del argv

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    save_dir = os.path.abspath(
        "./pure_screenshots"
    )

    replay_dir = os.path.abspath(
        FLAGS.replay_dir
    )

    env = SC2MultimodalEnv(
    map_name=FLAGS.map,
    save_dir=save_dir,
    replay_dir=replay_dir,
    timestamp=timestamp,
    feature_dims=(256, 256),
    rgb_dims=(1280, 720),
    map_size=(64, 64)
    )

    agent = PureImageAgent(
        env.action_space
    )

    num_episodes = 1

    try:
        for episode in range(num_episodes):

            print(
                f"\nStarting Episode {episode + 1}"
            )

            total_reward = run_episode(
                env,
                agent
            )

            print(
                f"Episode {episode + 1} finished. "
                f"Total Reward: {total_reward}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
