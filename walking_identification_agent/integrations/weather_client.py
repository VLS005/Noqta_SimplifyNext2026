"""Weather lookup. Swap the body of get_weather() for a real weather API call
(e.g. NEA/data.gov.sg for Singapore, or OpenWeatherMap) before the final demo
if you want live weather; mocked is fine for the scripted demo route.

Returns a severity alongside condition/raining, mirroring the obstruction
contract shape - placeholder values for now (fake data), ready to plug into
a real weather API's own severity/alert-level field later."""


def get_weather(lat: float, lon: float) -> dict:
    return {"raining": False, "condition": "clear", "severity": "low"}
