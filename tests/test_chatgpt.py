# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from openai import AsyncOpenAI

from trading_r1.providers import ModelProfile, get_profile, provider_for
from trading_r1.service.app import create_app
from trading_r1.state import chatgpt

CLIENT = "oaiapp_example"
MODEL = "gpt-example"


class ChatGPTLoginTest(unittest.IsolatedAsyncioTestCase):
    async def test_callback_refresh_model_picker_and_disconnect(self) -> None:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        requests: list[tuple[str, dict[str, str]]] = []

        def post(url, *, data, timeout):
            requests.append((url, data))
            request = httpx.Request("POST", url)
            if url == chatgpt.REVOKE:
                return httpx.Response(200, request=request)
            if data["grant_type"] == "refresh_token":
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "access_token": "refreshed-access",
                        "refresh_token": "rotated-refresh",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    },
                )
            now = int(time.time())
            identity = jwt.encode(
                {
                    "iss": chatgpt.ISSUER,
                    "aud": CLIENT,
                    "sub": "example-user",
                    "nonce": nonce,
                    "iat": now,
                    "exp": now + 300,
                    "email": "test@example.invalid",
                },
                private_key,
                algorithm="RS256",
                headers={"kid": "test"},
            )
            return httpx.Response(
                200,
                request=request,
                json={
                    "access_token": "test-access",
                    "refresh_token": "test-refresh",
                    "id_token": identity,
                    "expires_in": 15,
                    "token_type": "Bearer",
                    "scope": chatgpt.SCOPES,
                },
            )

        def models(url, *, headers, timeout):
            self.assertEqual(url, f"{chatgpt.RESOURCE}/models")
            self.assertEqual(headers, {"Authorization": "Bearer refreshed-access"})
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={
                    "models": [
                        {"slug": MODEL, "display_name": "Example GPT", "visibility": "list"},
                        {"slug": "hidden", "visibility": "hide"},
                    ]
                },
            )

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(
                os.environ, {"TR1_HOME": temp, "TR1_API_TOKEN": "test-local", "TR1_API_AUTH": ""}
            ),
            patch.object(chatgpt.httpx, "post", side_effect=post),
            patch.object(chatgpt.httpx, "get", side_effect=models),
            patch.object(chatgpt.jwt, "PyJWKClient") as jwks,
        ):
            jwks.return_value.get_signing_key_from_jwt.return_value = SimpleNamespace(
                key=private_key.public_key()
            )
            app = create_app()
            transport = httpx.ASGITransport(app=app)
            async with (
                httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as api,
                httpx.AsyncClient(trust_env=False) as browser,
            ):
                self.assertEqual((await api.get("/chatgpt")).status_code, 401)
                api.headers["Authorization"] = "Bearer test-local"
                begin = (await api.post("/chatgpt/login", json={})).json()
                flow = parse_qs(urlsplit(begin["url"]).query)
                callback = flow["redirect_uri"][0]
                nonce = flow["nonce"][0]
                state = flow["state"][0]
                self.assertEqual(flow["client_id"], [chatgpt.DYNAMIC_CLIENT])
                self.assertEqual(flow["code_challenge_method"], ["S256"])

                rejected = await browser.get(callback, params={"state": "unrelated", "code": "bad"})
                self.assertEqual(rejected.status_code, 400)
                self.assertEqual(requests, [])
                redirect = await browser.get(
                    callback, params={"state": state, "code": "test-code", "client_id": CLIENT}
                )
                self.assertEqual(redirect.status_code, 200)
                await asyncio.wait_for(app.state.chatgpt_login.task, timeout=5)
                result = (await api.get(f"/chatgpt/login/{begin['id']}")).json()
                self.assertEqual(result, {"status": "complete", "account": CLIENT, "sharing": True})
                verifier = requests[0][1]["code_verifier"]
                challenge = (
                    base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                    .decode()
                    .rstrip("=")
                )
                self.assertEqual(flow["code_challenge"], [challenge])
                self.assertEqual(requests[0][1]["redirect_uri"], callback)
                permissions = stat.S_IMODE((Path(temp) / chatgpt.FILE).stat().st_mode)
                self.assertEqual(permissions, 0o600)

                endpoint = f"/chatgpt/accounts/{CLIENT}"
                catalog = await api.get(f"{endpoint}/models")
                self.assertEqual(catalog.json(), [{"slug": MODEL, "name": "Example GPT"}])
                selected = await api.put(f"{endpoint}/model", json={"model": MODEL})
                self.assertEqual(
                    selected.json(),
                    {
                        "active": CLIENT,
                        "accounts": [
                            {
                                "id": CLIENT,
                                "email": "test@example.invalid",
                                "connected": True,
                                "sharing": True,
                                "model": MODEL,
                            }
                        ],
                    },
                )
                self.assertEqual(
                    [data["grant_type"] for _, data in requests],
                    ["authorization_code", "refresh_token"],
                )
                profile = get_profile(chatgpt.PROFILE)
                self.assertEqual(profile.model, MODEL)
                self.assertEqual(profile.reader_profile, profile.name)
                health = await api.get("/health", params={"profile": profile.name})
                self.assertEqual(health.json()["status"], "ok")

                with patch.dict(os.environ, {"TR1_API_AUTH": "off"}):
                    self.assertEqual((await api.get("/chatgpt")).status_code, 403)
                    self.assertIsNone(get_profile(chatgpt.PROFILE).oauth_account)

                disconnected = await api.delete(f"{endpoint}/session")
                self.assertTrue(disconnected.json()["revoked"])
                self.assertEqual(requests[-1][1]["token"], "rotated-refresh")
                self.assertNotIn("refreshed-access", (Path(temp) / chatgpt.FILE).read_text())
                self.assertFalse((await api.get("/chatgpt")).json()["accounts"][0]["connected"])
                reconnect = (await api.post("/chatgpt/login", json={"account": CLIENT})).json()
                returning = parse_qs(urlsplit(reconnect["url"]).query)
                self.assertEqual(returning["client_id"], [CLIENT])
                self.assertEqual(returning["ext_agent_host_id"], flow["ext_agent_host_id"])
                canceled = await api.delete(f"/chatgpt/login/{reconnect['id']}")
                self.assertEqual(canceled.json()["status"], "error")


class ChatGPTProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_roundtrip_and_structured_reader_use_streaming_responses(self) -> None:
        bodies = []
        tool_call = {
            "type": "function_call",
            "namespace": "trading_r1",
            "name": "visit",
            "call_id": "call_example",
            "arguments": "{}",
            "id": "fc_example",
        }
        message = {
            "type": "message",
            "role": "assistant",
            "id": "msg_example",
            "content": [{"type": "output_text", "text": '{"summary":"Done"}', "annotations": []}],
        }

        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), f"{chatgpt.RESOURCE}/responses")
            bodies.append(json.loads(request.content))
            output = [tool_call] if len(bodies) == 1 else [message]
            event = {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_example",
                    "object": "response",
                    "model": MODEL,
                    "status": "completed",
                    "created_at": 0,
                    "output": output,
                },
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(event)}\n\n",
            )

        profile = ModelProfile(
            name="chatgpt", model=MODEL, oauth_account=CLIENT, provider="chatgpt-responses"
        )
        async with AsyncOpenAI(
            api_key="mock",
            base_url=chatgpt.RESOURCE,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            provider = provider_for(profile, client)
            history = [{"role": "system", "content": "Answer with sources"}]
            turn = await provider.generate(history, tools=[{"name": "visit"}], max_tokens=1024)
            self.assertEqual(turn.tool_calls[0].name, "visit")
            await provider.generate(
                [
                    *history,
                    turn.to_message(),
                    {"role": "tool", "content": "Source text", "tool_call_id": "call_example"},
                ],
                max_tokens=1024,
            )
            result = await provider.complete(
                history,
                max_tokens=1024,
                response_schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            )
        self.assertEqual(result.text, '{"summary":"Done"}')
        for body in bodies:
            self.assertTrue(body["stream"])
            self.assertFalse(body["store"])
            self.assertNotIn("max_output_tokens", body)
            self.assertEqual(body["input"][0]["role"], "developer")
        self.assertEqual(bodies[0]["tools"][0]["type"], "namespace")
        self.assertEqual(bodies[1]["input"][1]["namespace"], "trading_r1")
        self.assertEqual(bodies[1]["input"][2]["call_id"], "call_example")
        self.assertEqual(bodies[2]["text"]["format"]["type"], "json_schema")
