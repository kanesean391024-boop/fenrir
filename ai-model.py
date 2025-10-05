import json
import logging
import re
import requests
from typing import List, Dict, Optional

from tools import tool_schema, call_tool

from runtime_utils import (
    create_object_logger,
    generate_with_watchdog,
    strip_think_markup,
    WATCHDOG_TRACKER,
)

logger = logging.getLogger(__name__)


class AIModel:
    """A single AI agent powered by an Ollama model."""

    def __init__(
        self,
        name: str,
        model_id: str,
        topic_prompt: str,
        role_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        chat_style: Optional[str] = None,
        watchdog_timeout: int | None = 900,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.debug(
            "Entering AIModel.__init__ with name=%s model_id=%s topic_prompt=%s role_prompt=%s temperature=%s max_tokens=%s chat_style=%s watchdog_timeout=%s system_prompt=%s",
            name,
            model_id,
            topic_prompt,
            role_prompt,
            temperature,
            max_tokens,
            chat_style,
            watchdog_timeout,
            system_prompt,
        )
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.watchdog_timeout = watchdog_timeout

        self.logger.info(
            "Initialized AIModel for %s using %s", self.name, self.model_id
        )

        self.role_prompt = role_prompt
        self.topic_prompt = topic_prompt

        self.format_instructions = (
            "The messages you receive are in the following format: the chat "
            "log begins with '=====Chat Begins=====' and ends with '=====Chat "
            "Ends====='. Each message is separated by a line of hyphens."
        )

        parts = []
        if chat_style:
            parts.append(f"Use a {chat_style} tone.")
        self.base_prompt = "\n".join(parts)

        # system_prompt is optional and comes from config
        self.system_prompt = system_prompt

        self.logger.debug("Exiting AIModel.__init__")

    def build_prompt(self, chat_log: List[Dict[str, str]]) -> str:
        """Assemble a prompt from system prompt and chat history."""
        self.logger.debug("Entering build_prompt with chat_log=%s", chat_log)
        lines = ["=====Chat Begins====="]
        for entry in chat_log:
            message = entry.get("message", "")
            lines.append(message)
            lines.append("-" * 80)
        lines.append("=====Chat Ends=====")

        prompt = "\n".join(lines)
        self.logger.debug("Built prompt of %d characters", len(prompt))
        self.logger.debug("Exiting build_prompt")
        return prompt

    def generate_response(
        self,
        chat_log: List[Dict[str, str]],
        *,
        override_model: Optional[str] = None,
        override_temperature: Optional[float] = None,
        system_text: Optional[str] = None,
    ) -> str:
        """Generate a response from the model using Ollama's API."""
        self.logger.debug("Entering generate_response with chat_log=%s", chat_log)
        prompt = self.build_prompt(chat_log)

        model_id = override_model or self.model_id
        temperature = (
            self.temperature if override_temperature is None else override_temperature
        )

        import json
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "AIModel.generate_response payload about to be sent:\n%s",
                json.dumps(
                    {
                        "model": model_id,
                        "prompt": prompt,
                        "temperature": temperature,
                    },
                    indent=2,
                ),
            )

        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {},
        }
        if system_text is None:
            system_parts = []
            if self.system_prompt:
                system_parts.append(self.system_prompt)
            role_topic = " ".join(
                [p for p in [self.topic_prompt, self.role_prompt] if p]
            )
            if role_topic:
                system_parts.append(role_topic)
            system_parts.append(self.format_instructions)
            if self.base_prompt:
                system_parts.append(self.base_prompt)
            if self.max_tokens is not None:
                system_parts.append(
                    f"Keep your response to less than {self.max_tokens} words. Otherwise, your response will be truncated."
                )
            if system_parts:
                system_text = "\n".join(system_parts)
        if system_text:
            payload["system"] = system_text

        self.logger.debug("Sending generation request")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        try:
            result_text = generate_with_watchdog(
                payload,
                base_timeout=self.watchdog_timeout,
                agent_name=self.name,
            )
        except requests.Timeout:
            raise
        result_text = strip_think_markup(result_text)
        self.logger.debug("Generated %d characters", len(result_text))
        self.logger.debug("Exiting generate_response")
        return result_text

    def generate_from_prompt(
        self,
        prompt: str,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        *,
        override_model: Optional[str] = None,
        override_temperature: Optional[float] = None,
        system_text: Optional[str] = None,
    ) -> str:
        """Generate a response from a custom prompt."""
        self.logger.debug(
            "Entering generate_from_prompt with num_ctx=%s num_predict=%s",
            num_ctx,
            num_predict,
        )
        model_id = override_model or self.model_id
        temperature = (
            self.temperature if override_temperature is None else override_temperature
        )
        import json
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "AIModel.generate_from_prompt payload about to be sent:\n%s",
                json.dumps(
                    {
                        "model": model_id,
                        "prompt": prompt,
                        "temperature": temperature,
                        "options": {
                            **({"num_ctx": num_ctx} if num_ctx is not None else {}),
                            **({"num_predict": num_predict} if num_predict is not None else {}),
                        },
                        **({"system": system_text} if system_text is not None else {}),
                    },
                    indent=2,
                ),
            )
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {},
        }
        if num_ctx is not None:
            payload.setdefault("options", {})["num_ctx"] = num_ctx
        # num_predict is accepted for compatibility but intentionally ignored
        if system_text is None:
            system_parts = []
            if self.system_prompt:
                system_parts.append(self.system_prompt)
            role_topic = " ".join(
                [p for p in [self.topic_prompt, self.role_prompt] if p]
            )
            if role_topic:
                system_parts.append(role_topic)
            if self.format_instructions:
                system_parts.append(self.format_instructions)
            if self.base_prompt:
                system_parts.append(self.base_prompt)
            if self.max_tokens is not None:
                system_parts.append(
                    f"Keep your response to less than {self.max_tokens} words. Otherwise, your response will be truncated."
                )
            if system_parts:
                system_text = "\n".join(system_parts)
        if system_text:
            payload["system"] = system_text
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        try:
            result_text = generate_with_watchdog(
                payload,
                base_timeout=self.watchdog_timeout,
                agent_name=self.name,
            )
        except requests.Timeout:
            raise
        result_text = strip_think_markup(result_text)
        self.logger.debug("Generated %d characters", len(result_text))
        self.logger.debug("Exiting generate_from_prompt")
        return result_text

    def chat_completion(
        self,
        messages: List[Dict[str, object]],
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Call Ollama chat API optionally with tools."""
        self.logger.debug(
            "Entering chat_completion with messages=%s tools=%s",
            messages,
            tools,
        )
        payload_messages = [m.copy() for m in messages]
        payload = {
            "model": self.model_id,
            "messages": payload_messages,
            "temperature": self.temperature,
            "options": {},
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        system_parts = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        role_topic = " ".join(
            [p for p in [self.topic_prompt, self.role_prompt] if p]
        )
        if role_topic:
            system_parts.append(role_topic)
        system_parts.append(self.format_instructions)
        if self.base_prompt:
            system_parts.append(self.base_prompt)
        if self.max_tokens is not None:
            system_parts.append(
                f"Keep your response to less than {self.max_tokens} words. Otherwise, your response will be truncated."
            )
        if system_parts:
            system_text = "\n".join(system_parts)
            payload["system"] = system_text
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        try:
            result_text = generate_with_watchdog(
                payload,
                base_timeout=self.watchdog_timeout,
                agent_name=self.name,
            )
        except requests.Timeout:
            raise
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            self.logger.error("Invalid JSON from Ollama: %s", exc)
            raise RuntimeError("Invalid JSON from Ollama") from exc
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = strip_think_markup(content)
        self.logger.debug("Exiting chat_completion")
        return data


class Agent:
    """Base class for all agents."""

    def __init__(
        self,
        name: str,
        model_name: str,
        role_prompt: str,
        config: Dict[str, Optional[str]],
        groups: Optional[List[str]] = None,
        groups_in: Optional[List[str]] = None,
        groups_out: Optional[List[str]] = None,
        allow_self_consume: bool = False,
    ) -> None:
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.debug(
            "Entering Agent.__init__ with name=%s model_name=%s role_prompt=%s groups=%s groups_in=%s groups_out=%s allow_self_consume=%s",
            name,
            model_name,
            role_prompt,
            groups,
            groups_in,
            groups_out,
            allow_self_consume,
        )
        self.name = name
        self.model_name = model_name
        self.role_prompt = role_prompt
        groups_in_set = set(groups_in or [])
        groups_out_set = set(groups_out or [])
        if groups:
            g_set = set(groups)
            groups_in_set |= g_set
            groups_out_set |= g_set
        if not groups_in_set:
            raise ValueError(f"Agent {self.name} missing groups_in")
        if not groups_out_set:
            raise ValueError(f"Agent {self.name} missing groups_out")
        self.groups_in = groups_in_set
        self.groups_out = groups_out_set
        self.groups = sorted(groups_in_set | groups_out_set)
        self.allow_self_consume = allow_self_consume
        self.active = True
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized agent %s", self.name)

        max_tok_val = config.get("max_tokens")
        max_tok = int(max_tok_val) if max_tok_val is not None else None
        self.model = AIModel(
            name=name,
            model_id=model_name,
            topic_prompt=config.get("topic_prompt", ""),
            role_prompt=role_prompt,
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=max_tok,
            chat_style=config.get("chat_style"),
            watchdog_timeout=int(config.get("watchdog_timeout", 900)),
            system_prompt=config.get("system_prompt"),
        )

        self.logger.debug("Exiting Agent.__init__")

    def step(self, context: List[Dict[str, str]]):
        self.logger.debug("Entering Agent.step with context=%s", context)
        self.logger.debug("Exiting Agent.step")
        raise NotImplementedError


class Tracer(Agent):
    """Lightweight classifier determining if a user's message was answered."""

    TRACE_INSTRUCTIONS = (
        "You are an internal Tracer process (not speaking to a human). "
        "You will be shown a message received from the humans and a message that "
        "was sent *to* the humans. Decide if the sent message sufficiently "
        "answers / responds to the received message. Reply ONLY 'Yes' or 'No'."
    )

    def step(self, context: List[Dict[str, str]]) -> str:  # pragma: no cover - defensive
        """Tracer does not engage in conversation."""
        self.logger.debug("Tracer.step called unexpectedly; ignoring.")
        return ""

    def _judge_pair(self, user_msg: str, sent_msg: str) -> bool:
        """Return True if sent_msg answers user_msg."""
        import json

        lines = [
            "Below is a message received from the humans and a message sent to the humans. Does the sent message respond to the received message? Only answer yes or no:",
            "-----------------",
            f"Received Message: {user_msg}",
            "-----------------",
            f"Sent Message: {sent_msg}",
        ]
        prompt = "\n".join(lines)
        wc = len(prompt.split())

        payload = {
            "model": self.model.model_id,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "options": {
                "num_ctx": wc,
                "num_predict": 3,
                "top_k": 1,
                "top_p": 0,
            },
            "system": self.TRACE_INSTRUCTIONS,
        }
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Tracer _judge_pair payload:\n%s", json.dumps(payload, indent=2))

        text = generate_with_watchdog(
            payload,
            base_timeout=self.model.watchdog_timeout,
            agent_name=self.name,
        )
        text = strip_think_markup(text)
        cleaned = re.sub(r"[^a-zA-Z]", "", text).lower()
        self.logger.info(
            "Tracer._judge_pair: raw=%r → cleaned=%r → verdict=%s",
            text,
            cleaned,
            "yes" in cleaned,
        )
        return "yes" in cleaned

    def is_answered(self, user_message: str, outputs: List[str]) -> bool:
        """Return True iff any sent message answers user_message."""
        if not outputs:
            return False
        result = any(self._judge_pair(user_message, out) for out in outputs)
        self.logger.info(
            "Tracer.is_answered: user=%r, outputs=%s → %s",
            user_message,
            outputs,
            result,
        )
        return result


class Ruminator(Agent):
    """Regular discussion participant."""

    def step(self, context: List[Dict[str, str]]) -> str:
        self.logger.debug("Entering Ruminator.step with context=%s", context)
        self.logger.info("Generating response")
        reply = self.model.generate_response(context)
        self.logger.info("Ruminator.step: generated reply of length %d", len(reply))
        self.logger.debug("Response length %d", len(reply))
        self.logger.debug("Exiting Ruminator.step")
        return reply


class Ponderer(Ruminator):
    """Ruminator that reduces boredom when active."""
    pass


class Doubter(Ruminator):
    """Ruminator that decreases certainty when active."""
    pass


class ToolAgent(Agent):
    """Agent capable of using tools via the Ollama API."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.debug("Entering ToolAgent.__init__ with args=%s kwargs=%s", args, kwargs)
        super().__init__(*args, **kwargs)
        self.tools = tool_schema()
        self.logger.debug("Exiting ToolAgent.__init__")

    def step(self, context: List[Dict[str, str]]) -> str:
        self.logger.debug("Entering ToolAgent.step with context=%s", context)
        messages: List[Dict[str, object]] = []
        for entry in context:
            sender = entry.get("sender", "")
            content = entry.get("message", "")
            role = "assistant" if sender == self.name else "user"
            messages.append({"role": role, "content": content})

        while True:
            resp = self.model.chat_completion(messages, tools=self.tools)
            msg = resp.get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                return content

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                call_id = call.get("id")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                result = call_tool(str(name), args)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
        self.logger.debug("Exiting ToolAgent.step")


class Archivist(Agent):
    """Non-participating summarizer and archivist."""

    def step(self, full_context: List[Dict[str, str]]) -> str:
        """Archive transcript and return compressed summary."""
        self.logger.debug(
            "Entering Archivist.step with full_context=%s", full_context
        )
        from datetime import datetime
        import os

        self.logger.info("Archiving full transcript")

        lines = []
        prompt_lines = []
        for entry in full_context:
            sender = entry.get("sender", "")
            message = entry.get("message", "")
            timestamp = entry.get("timestamp", "")
            lines.append(f"[{timestamp}] {sender}: {message}")
            prompt_lines.append(message)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            os.makedirs("archive", exist_ok=True)
            fname = os.path.join("archive", f"{ts}_full.txt")
            with open(fname, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        except OSError as exc:
            self.logger.error("Failed to archive conversation: %s", exc)

        prompt_lines.append(
            "Your job is to summarize the above chat. Keep as much of the meaning of the conversation as you can. You are not responding to anyone, so do not actually speak directly to anyone. Simply summarize the chat."
        )
        prompt = "\n".join(prompt_lines)
        word_count = len(prompt.split())

        self.logger.info(
            "Archivist.step: generating summary with num_ctx=%d", word_count
        )

        summary = self.model.generate_from_prompt(prompt, num_ctx=word_count)

        self.logger.info("Summary generated")
        self.logger.debug("Exiting Archivist.step")
        return summary


class Listener(Agent):
    """Agent that monitors user questions and notifies other AIs."""

    PROMPT_INSTRUCTIONS = (
        "You are a Listener AI speaking to other AIs. "
        "The user's message displays under -----Message from User----- "
        "Gently remind the other AIs that the user sent a message and restate the message in your own words."
    )

    def prompt_ais(self, transcript: str, message: str) -> str:
        """Ask other AIs to answer the user's question."""
        lines = ["-----Message from User-----"]
        lines.append(message)
        lines.append("-----Your Instructions-----")
        lines.append(self.PROMPT_INSTRUCTIONS)
        prompt = "\n".join(lines)
        self.logger.info("Listener.prompt_ais: sending prompt to other AIs")
        reply = self.model.generate_from_prompt(
            prompt,
            system_text="",
        )
        self.logger.info("Listener.prompt_ais: got reply of length %d", len(reply))
        return reply

    def step(self, context: List[Dict[str, str]]) -> str:
        """Generate a notice to other AIs about the most recent user message."""
        self.logger.debug("Entering Listener.step with context=%s", context)
        message = ""
        for entry in reversed(context):
            if entry.get("sender") in {"Human", "System"}:
                message = entry.get("message", "")
                break
        lines = ["-----Message from User-----", message, "-----Your Instructions-----", self.PROMPT_INSTRUCTIONS]
        prompt = "\n".join(lines)
        self.logger.info("Listener.step: sending prompt to model")
        reply = self.model.generate_from_prompt(prompt, system_text="")
        self.logger.info("Listener.step: got reply of length %d", len(reply))
        self.logger.debug("Exiting Listener.step")
        return reply


class Speaker(Agent):
    """Agent responsible for communicating with humans."""

    def step(self, context: List[Dict[str, str]]) -> str:
        """Generate a response intended for the outside world."""
        self.logger.debug("Entering Speaker.step with context=%s", context)
        self.logger.info("Generating response")
        prompt = self.model.build_prompt(context)
        parts = []
        if self.model.system_prompt:
            parts.append(self.model.system_prompt)
        role_topic = " ".join(
            [p for p in [self.model.topic_prompt, self.model.role_prompt] if p]
        )
        if role_topic:
            parts.append(role_topic)
        parts.append("You are speaking to humans.")
        system_text = "\n".join(parts)
        self.logger.info("Speaker.step: calling generate_from_prompt")
        reply = self.model.generate_from_prompt(prompt, system_text=system_text)
        self.logger.info("Speaker.step: got reply of length %d", len(reply))
        self.logger.debug("Response length %d", len(reply))
        self.logger.debug("Exiting Speaker.step")
        return reply
