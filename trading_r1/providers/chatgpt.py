# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""The Responses API request contract for a ChatGPT plan OAuth grant."""

from __future__ import annotations

import asyncio
from typing import Any

from openai import APIStatusError, AsyncOpenAI, Omit

from trading_r1.providers.base import AssistantTurn, Completion, DeltaCallback
from trading_r1.providers.openai import (
    _INCLUDE,
    OpenAIProvider,
    _to_input,
    _to_tools,
    _turn_from_response,
)
from trading_r1.providers.usage import record_openai
from trading_r1.state import chatgpt

_NAMESPACE = "trading_r1"
_PLAN_ERRORS = {
    "subscription_sharing_usage_limit_exceeded": (
        "The ChatGPT plan usage limit was reached. See https://chatgpt.com/settings/usage."
    ),
    "subscription_sharing_user_not_eligible": (
        "ChatGPT plan usage is unavailable for this account or workspace."
    ),
    "subscription_sharing_unsupported_capability": (
        "The selected ChatGPT model does not support this request."
    ),
    "subscription_sharing_invalid_user": (
        "OpenAI could not validate the ChatGPT account. Check access or reconnect."
    ),
}
_TEMPORARY_PLAN_ERRORS = {
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_unavailable",
}


class _ChatGPTUnavailable(chatgpt.ChatGPTError):
    status_code = 503


class ChatGPTProvider(OpenAIProvider):
    def _input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = [dict(item) for item in _to_input(self.profile, messages)]
        for item in items:
            if item.get("role") == "system":
                item["role"] = "developer"
            if item.get("type") == "function_call":
                item.setdefault("namespace", _NAMESPACE)
        return items

    async def _stream(self, options: dict[str, Any], on_delta: DeltaCallback | None) -> Any:
        if not self.profile.oauth_account or not self.profile.model:
            raise chatgpt.ChatGPTError(
                "Connect a ChatGPT account and choose a model in Trading-R1."
            )
        if self.client is not None:
            return await self._read_stream(self.client, options, on_delta)
        token = await asyncio.to_thread(chatgpt.access_token, self.profile.oauth_account)
        async with AsyncOpenAI(
            api_key=token,
            base_url=chatgpt.RESOURCE,
            default_headers={"OpenAI-Organization": Omit(), "OpenAI-Project": Omit()},
            timeout=self.profile.timeout_s,
            max_retries=0,
        ) as client:
            try:
                return await self._read_stream(client, options, on_delta)
            except APIStatusError as err:
                message = _PLAN_ERRORS.get(err.code or "")
                if not message and err.status_code in (401, 403):
                    message = f"OpenAI refused the ChatGPT request (HTTP {err.status_code})."
                if message:
                    request_id = f" Request ID: {err.request_id}." if err.request_id else ""
                    raise chatgpt.ChatGPTError(message + request_id) from None
                raise

    async def _read_stream(
        self, client: AsyncOpenAI, options: dict[str, Any], on_delta: DeltaCallback | None
    ) -> Any:
        stream = await client.responses.create(
            model=self.profile.model, store=False, stream=True, **options
        )
        async with stream:
            async for event in stream:
                if event.type == "response.completed":
                    record_openai(self.profile.name, event.response)
                    return event.response
                if event.type == "response.incomplete":
                    record_openai(self.profile.name, event.response)
                    raise chatgpt.ChatGPTError("OpenAI returned an incomplete ChatGPT response.")
                if event.type in ("response.failed", "error"):
                    detail = event.response.error if event.type == "response.failed" else event
                    code = getattr(detail, "code", None)
                    if code in _TEMPORARY_PLAN_ERRORS:
                        raise _ChatGPTUnavailable("ChatGPT plan usage is temporarily unavailable.")
                    raise chatgpt.ChatGPTError(
                        _PLAN_ERRORS.get(code, "OpenAI could not complete the ChatGPT request.")
                    )
                if on_delta and event.type == "response.output_text.delta":
                    on_delta("content", event.delta)
                if on_delta and event.type == "response.reasoning_summary_text.delta":
                    on_delta("reasoning", event.delta)
        raise chatgpt.ChatGPTError("The ChatGPT response stream ended before completion.")

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int,
        on_delta: DeltaCallback | None = None,
    ) -> AssistantTurn:
        options: dict[str, Any] = {
            "input": self._input(messages),
            "reasoning": {"summary": "auto"},
            "include": _INCLUDE,
        }
        if functions := _to_tools(tools):
            options["tools"] = [
                {
                    "type": "namespace",
                    "name": _NAMESPACE,
                    "description": "Trading-R1 research, market data, and analysis tools",
                    "tools": functions,
                }
            ]
        response = await self._stream(options, on_delta)
        return _turn_from_response(self.profile, response)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion:
        options: dict[str, Any] = {"input": self._input(messages)}
        if response_schema is not None:
            options["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "trading_r1_result",
                    "schema": response_schema,
                    "strict": True,
                }
            }
        response = await self._stream(options, None)
        turn = _turn_from_response(self.profile, response)
        return Completion(turn.content, turn.finish_reason)
