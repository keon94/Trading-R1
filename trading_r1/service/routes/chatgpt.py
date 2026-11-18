# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""Local ChatGPT account management and the short-lived browser callback."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import re
import secrets
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from trading_r1.service.auth import auth_enabled
from trading_r1.state import chatgpt


def _require_service_auth() -> None:
    if not auth_enabled():
        raise HTTPException(403, "ChatGPT sign-in requires the authenticated local service.")


router = APIRouter(prefix="/chatgpt", dependencies=[Depends(_require_service_auth)])
_CALLBACK = "/auth/callback"
_LOGIN_TIMEOUT = 300
_ISSUED_CLIENT = re.compile(r"[A-Za-z0-9_-]{1,256}")


class Login:
    def __init__(self, host_id: str, account: chatgpt.Account | None) -> None:
        self.host_id = host_id
        self.account = account
        self.id = secrets.token_urlsafe(24)
        self.state = secrets.token_urlsafe(32)
        self.nonce = secrets.token_urlsafe(32)
        self.verifier = secrets.token_urlsafe(64)
        self.redirect_uri = ""
        self.server: asyncio.Server | None = None
        self.callback: asyncio.Future[dict[str, list[str]]] = (
            asyncio.get_running_loop().create_future()
        )
        self.task: asyncio.Task[None] | None = None
        self.result: dict[str, Any] = {"status": "pending"}

    async def start(self) -> str:
        self.server = await asyncio.start_server(
            self.receive, "127.0.0.1", 0, limit=16_384, backlog=8
        )
        port = self.server.sockets[0].getsockname()[1]
        self.redirect_uri = f"http://127.0.0.1:{port}{_CALLBACK}"
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self.verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        params = {
            "client_id": self.account.client_id if self.account else chatgpt.DYNAMIC_CLIENT,
            "ext_agent_host_id": self.host_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": chatgpt.SCOPES,
            "resource": chatgpt.RESOURCE,
            "state": self.state,
            "nonce": self.nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if self.account is None:
            params["agent_name_hint"] = "Trading-R1"
        elif not self.account.sharing:
            params["prompt"] = "consent"
        self.task = asyncio.create_task(self.finish())
        return f"{chatgpt.AUTHORIZE}?{urlencode(params)}"

    async def receive(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # The callback binds its own ephemeral loopback port. The regular service API
        # continues to require its local bearer token.
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=3)
            line, *headers = request.decode("ascii").split("\r\n")
            method, target, _ = line.split(" ")
            target_url = urlsplit(target)
            hosts = [h.split(":", 1)[1].strip() for h in headers if h.lower().startswith("host:")]
            if (
                method != "GET"
                or not target.startswith("/")
                or target_url.netloc
                or target_url.path != _CALLBACK
                or hosts != [urlsplit(self.redirect_uri).netloc]
            ):
                await self.reply(
                    writer, "404 Not Found", "This is the Trading-R1 sign-in callback."
                )
                return
            query = parse_qs(target_url.query, keep_blank_values=True, max_num_fields=32)
            state = query.get("state", [])
            if len(state) != 1 or not hmac.compare_digest(state[0].encode(), self.state.encode()):
                await self.reply(writer, "400 Bad Request", "Invalid Trading-R1 sign-in attempt.")
                return
            if self.callback.done() or any(len(values) != 1 for values in query.values()):
                await self.reply(writer, "400 Bad Request", "Invalid Trading-R1 sign-in callback.")
                return
            self.callback.set_result(query)
            await self.reply(writer, "200 OK", "Return to Trading-R1 to finish signing in.")
        except (
            ValueError,
            OSError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            pass
        finally:
            writer.close()

    @staticmethod
    async def reply(writer: asyncio.StreamWriter, status: str, message: str) -> None:
        body = message.encode()
        writer.write(
            (
                f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n"
                "Cache-Control: no-store\r\nReferrer-Policy: no-referrer\r\n"
                "X-Content-Type-Options: nosniff\r\n\r\n"
            ).encode()
            + body
        )
        await writer.drain()

    async def finish(self) -> None:
        try:
            query = await asyncio.wait_for(self.callback, timeout=_LOGIN_TIMEOUT)
            self.close_listener()
            if "error" in query:
                raise chatgpt.ChatGPTError("ChatGPT sign-in was declined or canceled.")
            code = query.get("code", [""])[0]
            client_id = query.get("client_id", [self.account.client_id if self.account else ""])[0]
            if (
                not code
                or not _ISSUED_CLIENT.fullmatch(client_id)
                or client_id == chatgpt.DYNAMIC_CLIENT
            ):
                raise chatgpt.ChatGPTError("OpenAI returned an incomplete ChatGPT registration.")
            if self.account and client_id != self.account.client_id:
                raise chatgpt.ChatGPTError("OpenAI returned a different ChatGPT registration.")
            if self.account is None:
                await asyncio.to_thread(chatgpt.register, client_id)
            tokens, identity = await asyncio.to_thread(
                chatgpt.exchange, client_id, code, self.verifier, self.redirect_uri, self.nonce
            )
            await asyncio.to_thread(chatgpt.save_login, client_id, tokens, identity)
            sharing = chatgpt.PLAN_SCOPE in (tokens.scope or "").split()
            self.result = {"status": "complete", "account": client_id, "sharing": sharing}
        except chatgpt.ChatGPTError as err:
            self.result = {"status": "error", "message": str(err)}
        except TimeoutError:
            self.result = {"status": "error", "message": "ChatGPT sign-in timed out. Try again."}
        except Exception:  # exception details may contain OAuth credentials
            self.result = {"status": "error", "message": "Could not finish ChatGPT sign-in."}
        finally:
            self.close_listener()

    def close_listener(self) -> None:
        if self.server is not None:
            self.server.close()


class LoginRequest(BaseModel):
    account: str | None = None


class SelectRequest(BaseModel):
    model: str


async def _run[T](fn: Callable[..., T], *args: Any) -> T:
    try:
        return await asyncio.to_thread(fn, *args)
    except chatgpt.ChatGPTError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err


def _login(request: Request, attempt: str) -> Login:
    login: Login | None = request.app.state.chatgpt_login
    if login is None or login.id != attempt:
        raise HTTPException(status_code=404, detail="That ChatGPT sign-in attempt has ended.")
    return login


@router.get("")
async def accounts() -> dict[str, Any]:
    return await _run(chatgpt.status)


@router.post("/login")
async def start_login(req: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    async with request.app.state.chatgpt_login_lock:
        current: Login | None = request.app.state.chatgpt_login
        if current and current.task and not current.task.done():
            raise HTTPException(409, "A ChatGPT sign-in is already in progress.")
        host_id, account = await _run(chatgpt.prepare_login, req.account)
        login = Login(host_id, account)
        try:
            url = await login.start()
        except OSError as err:
            raise HTTPException(
                503, "Could not start the local ChatGPT callback listener."
            ) from err
        request.app.state.chatgpt_login = login
    response.headers["Cache-Control"] = "no-store"
    return {"id": login.id, "url": url}


@router.get("/login/{attempt}")
async def login_status(attempt: str, request: Request) -> dict[str, Any]:
    return _login(request, attempt).result


@router.delete("/login/{attempt}")
async def cancel_login(attempt: str, request: Request) -> dict[str, Any]:
    login = _login(request, attempt)
    if login.task is not None and not login.callback.done():
        login.result = {"status": "error", "message": "ChatGPT sign-in was canceled."}
        login.close_listener()
        login.task.cancel()
        await asyncio.gather(login.task, return_exceptions=True)
    return login.result


@router.get("/accounts/{account}/models")
async def models(account: str) -> list[dict[str, str]]:
    return await _run(chatgpt.list_models, account)


@router.put("/accounts/{account}/model")
async def select_model(account: str, req: SelectRequest) -> dict[str, Any]:
    await _run(chatgpt.select_model, account, req.model)
    return await _run(chatgpt.status)


@router.delete("/accounts/{account}/session")
async def sign_out(account: str) -> dict[str, Any]:
    confirmed = await _run(chatgpt.sign_out, account)
    return {"revoked": confirmed}
