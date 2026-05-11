from typing import Any
from urllib.parse import quote_plus

import httpx

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        location=StringSchema("city or location to query"),
        units=StringSchema("Unit system", enum=["metric", "us"]),
        required=["location"],
    ),
    # {
    #     "text": "object",
    #     "property": {
    #         "location": {"type": "string", "description": "city or location to query"},
    #         "units": {
    #             "type": "string",
    #             "description": "Unit system",
    #             "enum": ["metric", "us"]
    #         },
    #     },
    #     "required": ["location"],
    # }
)
class WeatherTool(Tool):
    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get current weather for a city or location."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, location: str, units: str = "metric", **_: Any) -> str:
        location = location.strip()
        if not location:
            return "Error: location is required"
        query = quote_plus(location)
        units_flag = "u" if units == "us" else "m"
        url = f"https://wttr.in/{query}?format=3&{units_flag}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        result = response.text.strip()
        if not result:
            return f"Error: weather service returned empty response for {location}"
        return result
