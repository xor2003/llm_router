import json
import re
from typing import Any


def generate_xml_tool_definitions(tools: list[dict[str, Any]]) -> str:
    """Converts a list of OpenAI-style tool definitions (JSON) into
    the XML format required by the RooCode-style prompt.
    """
    xml_definitions = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name")
        description = func.get("description")
        if not name or not description:
            continue

        params = func.get("parameters", {}).get("properties", {})
        required_params = func.get("parameters", {}).get("required", [])

        xml_def = f"## {name}\nDescription: {description}\n"

        if params:
            xml_def += "Parameters:\n"
            for param_name, param_details in params.items():
                is_required = "(required)" if param_name in required_params else "(optional)"
                param_desc = param_details.get("description", "")
                xml_def += f"- {param_name}: {is_required} {param_desc}\n"

        # Create usage example
        usage_example = f"<usage>\n<{name}>\n"
        for param_name in params.keys():
            usage_example += f"<{param_name}>Your {param_name} here</{param_name}>\n"
        usage_example += f"</{name}>\n</usage>"

        xml_definitions.append(xml_def + usage_example)

    return "\n\n".join(xml_definitions)


def parse_xml_tool_call(text: str) -> dict[str, Any] | None:
    """Parses the XML tool call from the LLM's response text.
    Returns a dictionary with tool_name and parameters, or None.
    """
    # Find all top-level XML tags
    tool_matches = re.findall(r"<(\w+?)>(.*?)</\1>", text, re.DOTALL)
    if not tool_matches:
        return None

    # Find the first non-internal tag (ignore <thinking> etc.)
    for tool_name, inner_xml in tool_matches:
        if tool_name not in ["thinking", "reasoning"]:
            inner_xml = inner_xml.strip()
            parameters = {}

            # Extract parameters from inner XML
            param_matches = re.findall(r"<(\w+?)>(.*?)</\1>", inner_xml, re.DOTALL)
            for param_name, param_value in param_matches:
                # If 'arguments' contains JSON, parse it
                if param_name == "arguments":
                    try:
                        parameters.update(json.loads(param_value.strip()))
                    except json.JSONDecodeError:
                        parameters[param_name] = param_value.strip()
                else:
                    parameters[param_name] = param_value.strip()

            return {
                "type": "tool_call",
                "tool_name": tool_name,
                "parameters": parameters,
            }

    return None
