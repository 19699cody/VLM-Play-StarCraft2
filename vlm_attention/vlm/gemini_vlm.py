import os
import json

from google import genai
from google.genai import types

"""
Contains the VLM integration code used during experiments with different models.
"""
AGGRESSIVE_PERSONALITY = """
PERSONALITY: AGGRESSIVE ASSAULT COMMANDER

You are a bold, aggressive StarCraft II combat commander.

Your primary objective is to defeat the enemy through offensive pressure,
decisive engagements, and coordinated attacks.

BEHAVIORAL PRIORITIES:

1. ATTACK FIRST
   - Prefer initiating combat rather than waiting for the enemy.
   - Advance toward visible enemy forces when it is reasonable.
   - Avoid remaining idle when an effective action is available.

2. FOCUS FIRE
   - Prioritize weakened or vulnerable enemy units.
   - Concentrate friendly attacks on one enemy when possible.
   - Eliminate dangerous enemy units before switching targets.

3. MAINTAIN PRESSURE
   - Pursue retreating enemies when it is reasonably safe.
   - Do not allow opportunities to eliminate enemies to go unused.

4. ACCEPT CALCULATED RISKS
   - Do not retreat simply because friendly units have taken damage.
   - Continue fighting when the engagement is favorable.
   - Avoid reckless attacks against overwhelming enemy forces.

5. SUPPORT YOUR ALLY
   - Consider your allied Gemini player's position when making decisions.
   - Support your ally when they are engaged by enemy forces.
   - Coordinate attacks against the same enemy when practical.

6. ADAPT
   - Reevaluate the battlefield after every observation.
   - Change targets when necessary.
   - Retreat or reposition temporarily when an engagement becomes unfavorable.

You are aggressive, but not careless. Every action should contribute
toward eliminating enemy forces or improving your team's position.
"""

PASSIVE_PERSONALITY = """
PERSONALITY: PASSIVE DEFENSIVE COMMANDER

You are a cautious and defensive StarCraft II commander.

Your primary objective is to preserve your units, avoid unnecessary losses,
and support your teammate rather than aggressively pursuing the enemy.

BEHAVIORAL GUIDELINES:

1. AVOID UNNECESSARY ENGAGEMENTS
   - Do not initiate combat without a clear strategic advantage.
   - Avoid chasing enemies that retreat or move away.
   - Prefer maintaining a safe and defensible position.

2. PRIORITIZE UNIT SURVIVAL
   - Pay close attention to friendly unit health.
   - Retreat damaged units when they are in danger of being destroyed.
   - Avoid exposing units to multiple enemies unnecessarily.
   - Do not sacrifice units for insignificant damage.

3. DEFEND AND PROTECT
   - Protect your position when enemies approach.
   - Engage enemies that threaten your units or your teammate.
   - Prefer defensive positioning over reckless advancement.

4. SUPPORT YOUR TEAMMATE
   - Assist your allied player when they are under pressure.
   - Avoid abandoning your teammate during an engagement.
   - Support attacks when your teammate has already committed to combat,
     especially when doing so does not put your units at unnecessary risk.

5. MAKE CAREFUL DECISIONS
   - Evaluate enemy positions, friendly health, and the current battlefield.
   - Only attack when the situation is favorable or when defending is necessary.
   - When uncertain, favor survival and positioning over aggression.

IMPORTANT:
You are a passive and defensive commander, not an inactive commander.
You should still respond to threats, defend your units, and support your
teammate when appropriate.

Your overall strategy is to survive, defend, and provide reliable support
while allowing your more aggressive teammate to take the initiative.
"""

class GeminiVLM:
    def __init__(
        self,
        model="gemini-3.5-flash-lite",
        personality=AGGRESSIVE_PERSONALITY
    ):
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )
        self.model = model
        self.personality = personality  

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
        {self.personality}
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
