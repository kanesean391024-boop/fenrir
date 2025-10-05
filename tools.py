import json
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class Tools:
    """Collection of callable tools for agents."""

    @staticmethod
    def get_current_weather(city: str) -> str:
        logger.debug("Entering get_current_weather city=%s", city)
        """Return a placeholder weather report for the given city."""
        result = f"It is always sunny in {city}."
        logger.debug("Exiting get_current_weather")
        return result

    @staticmethod
    def search_files(keyword: str) -> str:
        logger.debug("Entering search_files keyword=%s", keyword)
        """Search files in the current directory matching the keyword."""
        matches = []
        for root, _dirs, files in os.walk('.'):
            for fname in files:
                if keyword.lower() in fname.lower():
                    matches.append(os.path.join(root, fname))
        if not matches:
            result = f"No files found for '{keyword}'."
            logger.debug("Exiting search_files")
            return result
        result = ", ".join(matches)
        logger.debug("Exiting search_files")
        return result


def tool_schema() -> List[Dict[str, object]]:
    logger.debug("Entering tool_schema")
    """Return Ollama tool metadata for all available tools."""
    result = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search local files for a keyword",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Keyword to search for",
                        }
                    },
                    "required": ["keyword"],
                },
            },
        },
    ]
    logger.debug("Exiting tool_schema")
    return result


def tool_descriptions() -> str:
    logger.debug("Entering tool_descriptions")
    """Return a human readable list of available tools."""
    result = (
        "get_current_weather(city) - Get the current weather for a city.\n"
        "search_files(keyword) - Search local files for a keyword."
    )
    logger.debug("Exiting tool_descriptions")
    return result


def call_tool(name: str, args: Dict[str, object]) -> str:
    logger.debug("Entering call_tool name=%s args=%s", name, args)
    """Invoke the named tool with arguments and return its result."""
    if name == "get_current_weather":
        result = Tools.get_current_weather(**args)
        logger.debug("Exiting call_tool")
        return result
    if name == "search_files":
        result = Tools.search_files(**args)
        logger.debug("Exiting call_tool")
        return result
    logger.debug("Exiting call_tool with error")
    raise ValueError(f"Unknown tool {name}")
