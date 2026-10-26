from .gemini_vlm import GeminiVLM

"""
Tests Gemini responses and handles structured JSON commands.
"""


if __name__ == "__main__":

    screenshot_path = "./pure_screenshots/observation_0002.png"

    unit_info = [
        {
            "alliance": 1,
            "unit_name": "Stalker_3",
            "simplified_tag": 3,
            "health": 31.0,
            "max_health": 80.0,
            "position": (345, 50),
        },
        {
            "alliance": 4,
            "unit_name": "Zealot_1",
            "simplified_tag": 4,
            "health": 27.0,
            "max_health": 100.0,
            "position": (397, 12),
        },
        {
            "alliance": 4,
            "unit_name": "Zealot_2",
            "simplified_tag": 5,
            "health": 51.0,
            "max_health": 100.0,
            "position": (397, 25),
        },
        {
            "alliance": 4,
            "unit_name": "Zealot_3",
            "simplified_tag": 6,
            "health": 100.0,
            "max_health": 100.0,
            "position": (397, 40),
        },
    ]

    vlm = GeminiVLM()

    decision = vlm.decide(
        screenshot_path=screenshot_path,
        unit_info=unit_info,
    )

    print("\nPARSED GEMINI DECISION")
    print("=" * 60)
    print("Command:", decision["command"])
    print("Reason:", decision["reason"])
    print("=" * 60)