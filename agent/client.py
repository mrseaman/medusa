"""OpenAI API client with function-calling loop."""

import json
import logging

from openai import OpenAI

from .prompts import SYSTEM_PROMPT
from .tools import get_tool_schemas, dispatch
from .state import SessionState
from .models import ModelRegistry

logger = logging.getLogger(__name__)


class AgentClient:
    def __init__(self, state: SessionState, registry: ModelRegistry,
                 model: str = "gpt-4o"):
        self.state = state
        self.registry = registry
        self.model = model
        self.client = OpenAI()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tools = get_tool_schemas()

    def chat(self, user_message: str, on_tool_call=None) -> str:
        """Send a user message and return the assistant's final text response.

        Args:
            user_message: The user's input.
            on_tool_call: Optional callback(tool_name, tool_args, tool_result)
                          called each time a tool is executed.
        """
        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                parallel_tool_calls=True,
            )

            choice = response.choices[0]
            message = choice.message

            # Append the assistant message (may contain tool_calls)
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or ""

            # Execute each tool call and feed results back
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.debug(f"Tool call: {name}({args})")

                try:
                    result = dispatch(name, self.state, self.registry, **args)
                except Exception as e:
                    result = {"error": str(e)}

                result_str = json.dumps(result, default=str)

                if on_tool_call:
                    on_tool_call(name, args, result)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

    def reset_conversation(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
