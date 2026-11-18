# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""Private ChatGPT account registrations and renewable OAuth sessions."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import httpx
import jwt
from pydantic import BaseModel, Field, ValidationError

from trading_r1.host.instance import AUTH_OFF_ENV
from trading_r1.state.fileio import (
    MISSING,
    PrivateFileError,
    file_lock,
    read_private_json,
    write_json,
)
from trading_r1.state.paths import StateDirNotPrivateError, private_state_dir, state_dir

ISSUER = "https://auth.openai.com"
AUTHORIZE = f"{ISSUER}/api/accounts/authorize"
TOKEN = f"{ISSUER}/api/accounts/oauth/token"
JWKS = f"{ISSUER}/.well-known/jwks.json"
REVOKE = f"{ISSUER}/api/accounts/oauth/revoke"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"
DYNAMIC_CLIENT = "dynamic_agent_client"
_EXPIRED_GRANTS = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}
PROFILE = "chatgpt"
FILE = "chatgpt-auth.json"


class ChatGPTError(RuntimeError):
    """An account or OAuth error safe to show in the terminal."""


class OAuthTokens(BaseModel):
    access_token: str = Field(min_length=1, repr=False)
    refresh_token: str | None = Field(default=None, repr=False)
    id_token: str | None = Field(default=None, repr=False)
    scope: str | None = None
    token_type: str
    expires_in: int = Field(gt=0)


class Account(BaseModel):
    client_id: str
    subject: str | None = None
    email: str | None = None
    model: str | None = None
    access_token: str | None = Field(default=None, repr=False)
    refresh_token: str | None = Field(default=None, repr=False)
    id_token: str | None = Field(default=None, repr=False)
    expires_at: float = 0
    scopes: list[str] = Field(default_factory=list)

    @property
    def sharing(self) -> bool:
        return PLAN_SCOPE in self.scopes and bool(
            self.refresh_token or (self.access_token and self.expires_at > time.time())
        )

    def display(self) -> dict[str, Any]:
        return {
            "id": self.client_id,
            "email": self.email,
            "connected": bool(self.access_token),
            "sharing": self.sharing,
            "model": self.model,
        }

    def with_tokens(self, tokens: OAuthTokens, *, refresh: bool = False) -> Account:
        return self.model_copy(
            update={
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token or (self.refresh_token if refresh else None),
                "id_token": self.id_token if refresh else tokens.id_token,
                "expires_at": time.time() + tokens.expires_in,
                "scopes": self.scopes
                if refresh and tokens.scope is None
                else (tokens.scope or "").split(),
            }
        )

    def without_tokens(self) -> Account:
        return Account(
            client_id=self.client_id, subject=self.subject, email=self.email, model=self.model
        )


class Store(BaseModel):
    host_id: str | None = None
    active: str | None = None
    accounts: dict[str, Account] = Field(default_factory=dict)


def _load() -> Store:
    try:
        data = read_private_json(state_dir() / FILE)
        return Store() if data is MISSING else Store.model_validate(data)
    except (PrivateFileError, ValidationError) as err:
        raise ChatGPTError(f"Cannot read the private ChatGPT credentials file ({FILE}).") from err


def _save(store: Store) -> None:
    write_json(private_state_dir() / FILE, store.model_dump(mode="json"), private=True)


@contextmanager
def _locked() -> Iterator[Store]:
    try:
        with file_lock(private_state_dir() / "chatgpt-auth.lock"):
            yield _load()
    except (OSError, StateDirNotPrivateError) as err:
        raise ChatGPTError("Cannot update the private ChatGPT credentials file.") from err


def _account(store: Store, client_id: str) -> Account:
    if client_id not in store.accounts:
        raise ChatGPTError("That ChatGPT registration is no longer available.")
    return store.accounts[client_id]


def active_account() -> Account | None:
    if os.environ.get(AUTH_OFF_ENV, "").lower() == "off":
        return None
    store = _load()
    return store.accounts.get(store.active) if store.active else None


def status() -> dict[str, Any]:
    store = _load()
    return {
        "accounts": [account.display() for account in store.accounts.values()],
        "active": store.active,
    }


def prepare_login(client_id: str | None) -> tuple[str, Account | None]:
    with _locked() as store:
        account = _account(store, client_id) if client_id else None
        if not store.host_id:
            store.host_id = uuid4().urn
            _save(store)
        return store.host_id, account


def register(client_id: str) -> None:
    with _locked() as store:
        if client_id not in store.accounts:
            store.accounts[client_id] = Account(client_id=client_id)
            _save(store)


def save_login(client_id: str, tokens: OAuthTokens, identity: dict[str, Any]) -> None:
    with _locked() as store:
        current = _account(store, client_id)
        if current.subject and current.subject != identity["sub"]:
            raise ChatGPTError(
                "OpenAI returned a different ChatGPT account. Add it as a new account."
            )
        email = identity.get("email")
        updated = current.with_tokens(tokens).model_copy(
            update={
                "subject": identity["sub"],
                "email": email if isinstance(email, str) else current.email,
            }
        )
        store.accounts[client_id] = updated
        store.active = client_id
        _save(store)


class _GrantRejected(ChatGPTError):
    """The renewable session has expired or been revoked."""


def _token_request(data: dict[str, str]) -> OAuthTokens:
    try:
        response = httpx.post(TOKEN, data={**data, "resource": RESOURCE}, timeout=15)
        if response.status_code in (400, 401):
            error = response.json().get("error")
            if isinstance(error, str) and error in _EXPIRED_GRANTS:
                raise _GrantRejected("ChatGPT authorization expired or was revoked. Sign in again.")
        response.raise_for_status()
        tokens = OAuthTokens.model_validate(response.json())
    except (httpx.HTTPError, ValueError, AttributeError) as err:
        raise ChatGPTError("OpenAI could not issue a ChatGPT token. Try again.") from err
    if tokens.token_type.lower() != "bearer":
        raise ChatGPTError("OpenAI returned an unsupported ChatGPT token type.")
    return tokens


def exchange(
    client_id: str, code: str, verifier: str, redirect_uri: str, nonce: str
) -> tuple[OAuthTokens, dict[str, Any]]:
    tokens = _token_request(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        }
    )
    if not tokens.id_token:
        raise ChatGPTError("OpenAI did not return a ChatGPT identity token.")
    try:
        key = jwt.PyJWKClient(JWKS, timeout=10).get_signing_key_from_jwt(tokens.id_token)
        identity = jwt.decode(
            tokens.id_token,
            key.key,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=client_id,
            options={"require": ["iss", "aud", "sub", "exp", "iat", "nonce"]},
        )
        if identity["nonce"] != nonce or not identity["sub"]:
            raise jwt.InvalidTokenError("invalid identity")
        audience = identity["aud"]
        azp = identity.get("azp")
        if (
            isinstance(audience, list) and len(audience) > 1 or azp is not None
        ) and azp != client_id:
            raise jwt.InvalidTokenError("invalid authorized party")
    except (jwt.PyJWTError, ValueError) as err:
        raise ChatGPTError("Could not verify the ChatGPT identity returned by OpenAI.") from err
    return tokens, identity


def access_token(client_id: str) -> str:
    if os.environ.get(AUTH_OFF_ENV, "").lower() == "off":
        raise ChatGPTError("ChatGPT plan usage requires the authenticated local service.")
    # Hold the cross-process lock throughout refresh so the daemon and terminal backend
    # cannot both spend the same rotating refresh token.
    with _locked() as store:
        account = _account(store, client_id)
        if PLAN_SCOPE not in account.scopes:
            raise ChatGPTError("Use Continue with ChatGPT to enable ChatGPT plan usage.")
        refresh_window = 60 if account.refresh_token else 0
        if account.access_token and account.expires_at > time.time() + refresh_window:
            return account.access_token
        if not account.refresh_token:
            raise ChatGPTError("Your ChatGPT session expired. Sign in again.")
        try:
            tokens = _token_request(
                {
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": account.refresh_token,
                }
            )
        except _GrantRejected:
            store.accounts[client_id] = account.without_tokens()
            _save(store)
            raise
        updated = account.with_tokens(tokens, refresh=True)
        store.accounts[client_id] = updated
        _save(store)
        if PLAN_SCOPE not in updated.scopes:
            raise ChatGPTError("ChatGPT plan usage is no longer authorized. Sign in again.")
        return tokens.access_token


def list_models(client_id: str) -> list[dict[str, str]]:
    token = access_token(client_id)
    try:
        response = httpx.get(
            f"{RESOURCE}/models", headers={"Authorization": f"Bearer {token}"}, timeout=10
        )
        if response.status_code in (401, 403):
            raise ChatGPTError(
                "OpenAI refused the ChatGPT connection. Check access or sign in again."
            )
        response.raise_for_status()
        models = response.json()["models"]
        if not isinstance(models, list):
            raise ValueError("invalid model catalog")
        return [
            {"slug": item["slug"], "name": item.get("display_name") or item["slug"]}
            for item in models
            if isinstance(item, dict)
            and item.get("visibility") == "list"
            and isinstance(item.get("slug"), str)
            and item["slug"]
        ]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as err:
        raise ChatGPTError("Could not load models for this ChatGPT account.") from err


def select_model(client_id: str, model: str) -> None:
    if model not in {item["slug"] for item in list_models(client_id)}:
        raise ChatGPTError("That model is unavailable for this ChatGPT account.")
    with _locked() as store:
        account = _account(store, client_id)
        if not account.sharing:
            raise ChatGPTError("ChatGPT plan usage is no longer enabled.")
        store.accounts[client_id] = account.model_copy(update={"model": model})
        store.active = client_id
        _save(store)


def sign_out(client_id: str) -> bool:
    with _locked() as store:
        account = _account(store, client_id)
        confirmed = not account.access_token
        if account.refresh_token:
            confirmed = _revoke(account)
        store.accounts[client_id] = account.without_tokens()
        _save(store)
        return confirmed


def _revoke(account: Account) -> bool:
    for delay in (0, 0.5, 1):
        time.sleep(delay)
        try:
            response = httpx.post(
                REVOKE,
                data={
                    "token": account.refresh_token,
                    "token_type_hint": "refresh_token",
                    "client_id": account.client_id,
                },
                timeout=4,
            )
        except httpx.RequestError:
            continue
        if response.status_code < 500:
            return response.status_code == 200
    return False
