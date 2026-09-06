"""
Minimally-audible user prompting.

Swap this for the real haptic vibration pattern + low-volume bone-conduction
audio output. Keep prompts short and infrequent - see AgentConfig.alert_cooldown_sec.
"""


def prompt_user_minimally_audible(message: str) -> str:
    print(f"[HAPTIC/AUDIO PROMPT]: {message}")
    return "yes"  # mocked user confirmation for demo purposes
