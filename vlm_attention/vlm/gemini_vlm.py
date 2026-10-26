import os
import json

from google import genai
from google.genai import types

"""
Contains the VLM integration code used during experiments with different models.
"""

class GeminiVLM:
    def __init__(self, model="gemini-3.5-flash-lite"):
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )
        self.model = model

    def format_unit_info(self, unit_info):
        friendly_units = []
        enemy_units = []

        for unit in unit_info:
            unit_description = (
                f"{unit['unit_name']} "
                f"(tag={unit['simplified_tag']})\n"
                f"  Health: {unit['health']}/{unit['max_health']}\n"
                f"  Position: {unit['position']}"
            )

            if unit["alliance"] == 1:
                friendly_units.append(unit_description)

            elif unit["alliance"] == 4:
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

    def decide(self, screenshot_path, unit_info):

        with open(screenshot_path, "rb") as f:
            image_data = f.read()

        game_state = self.format_unit_info(unit_info)

        prompt = f"""
You are controlling a small StarCraft II micromanagement scenario.

Use BOTH:
1. The screenshot
2. The structured game state

The structured game state is authoritative for:
- unit names
- unit tags
- health
- positions

You may choose ONLY ONE command:

ATTACK <enemy>
MOVE <unit> <x,y>
RETREAT <unit>
HOLD

Rules:

- ATTACK <enemy>: all surviving friendly units attack the specified enemy.
- MOVE <unit> <x,y>: move the specified friendly unit to the
  specified grid location. x and y must each be integers from 1 to 10.
  Example:
  MOVE Stalker_1 7,5
- RETREAT <unit>: move the specified friendly unit away from the enemy.
- HOLD: take no action.
Choose the best action based on the current situation.

Return ONLY JSON in this format:

{{
    "command": "ATTACK <enemy>",
    "reason": "Brief explanation"
}}

GAME STATE:

{game_state}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/png",
                ),
                prompt,
            ],
        )

        result_text = response.text.strip()

        print("\nRAW GEMINI RESPONSE")
        print("=" * 60)
        print(result_text)
        print("=" * 60)

        # Gemini sometimes wraps JSON in Markdown code fences.
        if result_text.startswith("```"):
            result_text = result_text.replace("```json", "", 1)
            result_text = result_text.replace("```", "", 1)
            result_text = result_text.strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            raise ValueError(
                f"Gemini did not return valid JSON:\n{result_text}"
            )

        if "command" not in result:
            raise ValueError(
                f"Gemini response is missing 'command': {result}"
            )

        if "reason" not in result:
            result["reason"] = "No reason provided."

        return result