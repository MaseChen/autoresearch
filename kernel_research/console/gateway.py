"""Mac-local FastAPI Gateway for the Autoresearch Console SPA."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Mapping
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .protocol import strict_json_loads
from .protocol import (
    DraftV1,
    OperationReceiptV1,
    PreparedOperationV1,
)
from .local_store import ConsoleLocalStore
from .transport import AgentTransport


SESSION_COOKIE = "kr_console_session"
SESSION_IDLE_SECONDS = 12 * 60 * 60
MAX_HTTP_BODY_BYTES = 1024 * 1024
_HOST_RE = re.compile(r"^(?:127\.0\.0\.1|localhost)(?::[0-9]{1,5})?$")
_PAGE_CURSOR_RE = re.compile(r"^[A-Za-z0-9_-]{1,2048}$")


def _now() -> float:
    return time.monotonic()


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _envelope(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": str(uuid.uuid4()),
        "data": dict(data),
    }


def _snapshot_coordinate(snapshot: Mapping[str, Any]) -> tuple[str, str]:
    identity = snapshot.get("runtime_identity")
    cursor = snapshot.get("cursor")
    runtime_digest = (
        identity.get("runtime_identity_digest")
        if isinstance(identity, dict)
        else None
    )
    if not isinstance(runtime_digest, str) or not isinstance(cursor, dict):
        raise ValueError("snapshot has no pagination coordinate")
    cursor_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            cursor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return runtime_digest, cursor_digest


def _page_cursor(
    *, section: str, key: str, runtime_digest: str, cursor_digest: str
) -> str:
    encoded = json.dumps(
        {
            "schema_version": 1,
            "section": section,
            "key": key,
            "runtime_identity_digest": runtime_digest,
            "source_cursor_digest": cursor_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def _decode_page_cursor(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or not _PAGE_CURSOR_RE.fullmatch(value):
        raise ValueError("pagination cursor is invalid")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        selected = strict_json_loads(raw, max_bytes=2048)
    except (UnicodeError, binascii.Error, TypeError, ValueError) as exc:
        raise ValueError("pagination cursor is invalid") from exc
    fields = {
        "schema_version",
        "section",
        "key",
        "runtime_identity_digest",
        "source_cursor_digest",
    }
    if (
        not isinstance(selected, dict)
        or set(selected) != fields
        or selected.get("schema_version") != 1
        or any(not isinstance(selected.get(name), str) for name in fields - {"schema_version"})
    ):
        raise ValueError("pagination cursor fields do not match")
    return selected


def _paginate(
    snapshot: Mapping[str, Any],
    *,
    section: str,
    items: list[Mapping[str, Any]],
    keys: list[str],
    limit: int,
    after: str | None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("page limit must be between 1 and 100")
    if len(items) != len(keys) or len(set(keys)) != len(keys):
        raise ValueError("snapshot page keys are not unique")
    runtime_digest, cursor_digest = _snapshot_coordinate(snapshot)
    start = 0
    if after is not None:
        selected = _decode_page_cursor(after)
        if (
            selected["section"] != section
            or selected["runtime_identity_digest"] != runtime_digest
            or selected["source_cursor_digest"] != cursor_digest
        ):
            raise ValueError("pagination cursor belongs to another snapshot")
        try:
            start = keys.index(selected["key"]) + 1
        except ValueError as exc:
            raise ValueError("pagination cursor is outside the bounded source window") from exc
    page = items[start : start + limit]
    page_keys = keys[start : start + limit]
    next_after = None
    if page and start + len(page) < len(items):
        next_after = _page_cursor(
            section=section,
            key=page_keys[-1],
            runtime_digest=runtime_digest,
            cursor_digest=cursor_digest,
        )
    return page, {
        "limit": limit,
        "next_after": next_after,
        "runtime_identity_digest": runtime_digest,
        "source_cursor": snapshot["cursor"],
    }


async def _strict_request_json(request: Request) -> Mapping[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_HTTP_BODY_BYTES:
                raise HTTPException(status_code=413, detail="request body is too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
    raw = await request.body()
    if len(raw) > MAX_HTTP_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body is too large")
    try:
        value = strict_json_loads(raw, max_bytes=MAX_HTTP_BODY_BYTES)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    return value


@dataclass
class _Session:
    csrf_token: str
    last_seen: float


class SessionRegistry:
    def __init__(self, bootstrap_token: str | None = None) -> None:
        token = bootstrap_token or secrets.token_urlsafe(32)
        if not isinstance(token, str) or len(token) < 32 or not token.isascii():
            raise ValueError("bootstrap token is invalid")
        self.bootstrap_token = token
        self._bootstrap_digest = _hash_secret(token)
        self._bootstrap_consumed = False
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def exchange(self, token: object) -> tuple[str, str]:
        if not isinstance(token, str) or not token.isascii():
            raise ValueError("bootstrap token is invalid")
        with self._lock:
            if self._bootstrap_consumed or not secrets.compare_digest(
                _hash_secret(token), self._bootstrap_digest
            ):
                raise ValueError("bootstrap token is invalid or already consumed")
            self._bootstrap_consumed = True
            session_token = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
            self._sessions[_hash_secret(session_token)] = _Session(
                csrf_token=csrf_token, last_seen=_now()
            )
            return session_token, csrf_token

    def authorize(self, token: str | None) -> _Session:
        if not token or not token.isascii():
            raise ValueError("Console session is missing")
        key = _hash_secret(token)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                raise ValueError("Console session is invalid")
            now = _now()
            if now - session.last_seen > SESSION_IDLE_SECONDS:
                del self._sessions[key]
                raise ValueError("Console session expired")
            session.last_seen = now
            return session


class SnapshotCache:
    def __init__(self, transport: AgentTransport, *, ttl_seconds: float = 1.0) -> None:
        self.transport = transport
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._observed = 0.0
        self._snapshot: Mapping[str, Any] | None = None
        self._runtime_identity_digest: str | None = None

    def get(self, *, force: bool = False) -> Mapping[str, Any]:
        with self._lock:
            now = _now()
            if (
                not force
                and self._snapshot is not None
                and now - self._observed <= self.ttl_seconds
            ):
                return self._snapshot
            payload = self.transport.call("snapshot", {"limit": 100})
            snapshot = payload.get("snapshot")
            if not isinstance(snapshot, dict):
                raise RuntimeError("remote snapshot is missing")
            identity = snapshot.get("runtime_identity")
            digest = (
                identity.get("runtime_identity_digest")
                if isinstance(identity, dict)
                else None
            )
            if not isinstance(digest, str):
                raise RuntimeError("remote snapshot has no runtime identity")
            if self._runtime_identity_digest is None:
                self._runtime_identity_digest = digest
            elif self._runtime_identity_digest != digest:
                self._snapshot = None
                raise RuntimeError("remote runtime identity changed; restart handshake")
            self._snapshot = snapshot
            self._observed = now
            return snapshot


class SnapshotEventBroker:
    """One shared remote sampler fan-outs bounded SSE events to every tab."""

    def __init__(self, cache: SnapshotCache, *, poll_interval_ms: int) -> None:
        if not 1000 <= poll_interval_ms <= 30000:
            raise ValueError("SSE poll interval is outside its trusted bounds")
        self.cache = cache
        self.poll_interval = poll_interval_ms / 1000.0
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._task: asyncio.Task[None] | None = None
        self._last_event_id = ""
        self._last_snapshot_event = ""
        self._heartbeat_at = 0.0
        self._runtime_identity_digest: str | None = None

    @staticmethod
    def _snapshot_event(snapshot: Mapping[str, Any]) -> tuple[str, str]:
        cursor = snapshot.get("cursor", {})
        event_id = hashlib.sha256(
            json.dumps(cursor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = json.dumps(
            {"snapshot": snapshot},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return event_id, f"id: {event_id}\nevent: snapshot\ndata: {payload}\n\n"

    def _verify_runtime_identity(self, snapshot: Mapping[str, Any]) -> None:
        identity = snapshot.get("runtime_identity")
        digest = (
            identity.get("runtime_identity_digest")
            if isinstance(identity, dict)
            else None
        )
        if not isinstance(digest, str):
            raise RuntimeError("subscription snapshot has no runtime identity")
        if self._runtime_identity_digest is None:
            self._runtime_identity_digest = digest
        elif self._runtime_identity_digest != digest:
            self._last_event_id = ""
            self._last_snapshot_event = ""
            raise RuntimeError("subscription runtime identity changed")

    def _publish(self, event: str) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    async def _sample(self) -> None:
        subscription: Any = None
        try:
            while self._subscribers:
                try:
                    subscribe = getattr(self.cache.transport, "subscribe", None)
                    if callable(subscribe):
                        if subscription is None:
                            subscription = subscribe()
                        payload = await asyncio.to_thread(next, subscription)
                        snapshot = payload.get("snapshot")
                        if not isinstance(snapshot, dict):
                            raise RuntimeError("subscription snapshot is missing")
                    else:
                        snapshot = await asyncio.to_thread(
                            self.cache.get, force=True
                        )
                    self._verify_runtime_identity(snapshot)
                    event_id, event = self._snapshot_event(snapshot)
                    if event_id != self._last_event_id:
                        self._last_event_id = event_id
                        self._last_snapshot_event = event
                        self._heartbeat_at = _now()
                        self._publish(event)
                    elif _now() - self._heartbeat_at >= 15:
                        self._heartbeat_at = _now()
                        self._publish(
                            'event: heartbeat\ndata: {"schema_version":1}\n\n'
                        )
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    if subscription is not None:
                        try:
                            await asyncio.to_thread(subscription.close)
                        except BaseException:
                            pass
                        subscription = None
                    self._publish('event: reset\ndata: {"schema_version":1}\n\n')
                    await asyncio.sleep(self.poll_interval)
                else:
                    if not callable(subscribe):
                        await asyncio.sleep(self.poll_interval)
        finally:
            if subscription is not None:
                try:
                    await asyncio.to_thread(subscription.close)
                except BaseException:
                    pass
            self._task = None

    async def stream(self, request: Request) -> Any:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self._subscribers.add(queue)
        previous = request.headers.get("last-event-id", "")
        if self._last_snapshot_event and previous != self._last_event_id:
            queue.put_nowait(self._last_snapshot_event)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._sample())
        try:
            while not await request.is_disconnected():
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
        finally:
            self._subscribers.discard(queue)
            if not self._subscribers and self._task is not None:
                self._task.cancel()


@dataclass
class GatewayState:
    transport: AgentTransport
    static_dir: Path
    poll_interval_ms: int = 2000
    sessions: SessionRegistry | None = None
    local_store: ConsoleLocalStore | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.static_dir, Path):
            raise TypeError("static_dir must be a Path")
        selected = self.static_dir.resolve(strict=True)
        if selected != self.static_dir or not selected.is_dir():
            raise ValueError("Console static_dir must be a canonical directory")
        index = selected / "index.html"
        if not index.is_file() or index.is_symlink():
            raise ValueError("Console static_dir is missing index.html")
        if not 1000 <= self.poll_interval_ms <= 30000:
            raise ValueError("poll_interval_ms is outside its trusted bounds")
        if self.sessions is None:
            self.sessions = SessionRegistry()
        if self.local_store is None:
            self.local_store = ConsoleLocalStore(selected.parent / "local-data")
        self.cache = SnapshotCache(self.transport)


def create_app(state: GatewayState) -> FastAPI:
    app = FastAPI(
        title="Autoresearch Console Gateway",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )
    assert state.sessions is not None
    assert state.local_store is not None
    event_broker = SnapshotEventBroker(
        state.cache, poll_interval_ms=state.poll_interval_ms
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        host = request.headers.get("host", "")
        if not _HOST_RE.fullmatch(host):
            return JSONResponse(status_code=400, content={"detail": "invalid Host"})
        nonce = secrets.token_urlsafe(24)
        request.state.csp_nonce = nonce
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "; ".join(
            (
                "default-src 'none'",
                f"script-src 'nonce-{nonce}' 'strict-dynamic'",
                f"style-src 'self' 'nonce-{nonce}'",
                "connect-src 'self'",
                "worker-src 'self'",
                "img-src 'self' data:",
                "font-src 'self'",
                "frame-ancestors 'none'",
                "base-uri 'none'",
                "form-action 'self'",
                "object-src 'none'",
            )
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    def session(request: Request) -> _Session:
        try:
            return state.sessions.authorize(request.cookies.get(SESSION_COOKIE))
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def mutation_session(request: Request) -> _Session:
        selected = session(request)
        host = request.headers.get("host", "")
        expected_origin = f"{request.url.scheme}://{host}"
        if request.headers.get("origin") != expected_origin:
            raise HTTPException(status_code=403, detail="Origin validation failed")
        if request.headers.get("sec-fetch-site") not in {"same-origin", "none"}:
            raise HTTPException(status_code=403, detail="Fetch Metadata validation failed")
        csrf = request.headers.get("x-csrf-token", "")
        if not secrets.compare_digest(csrf, selected.csrf_token):
            raise HTTPException(status_code=403, detail="CSRF validation failed")
        return selected

    @app.post("/api/v1/session/bootstrap")
    async def bootstrap(request: Request) -> Response:
        value = await _strict_request_json(request)
        if set(value) != {"schema_version", "token"} or value.get("schema_version") != 1:
            raise HTTPException(status_code=400, detail="bootstrap contract mismatch")
        try:
            token, csrf = state.sessions.exchange(value["token"])
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        response = JSONResponse(_envelope({"csrf_token": csrf}))
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/",
            max_age=SESSION_IDLE_SECONDS,
        )
        return response

    @app.get("/api/v1/health")
    def health() -> Mapping[str, Any]:
        return _envelope({"status": "READY", "gateway": "LOCAL_ONLY"})

    def snapshot_value() -> Mapping[str, Any]:
        try:
            return state.cache.get()
        except BaseException as exc:
            raise HTTPException(
                status_code=503, detail=f"remote snapshot unavailable: {type(exc).__name__}"
            ) from exc

    def page_value(
        snapshot: Mapping[str, Any],
        *,
        section: str,
        items: list[Mapping[str, Any]],
        keys: list[str],
        limit: int,
        after: str | None,
    ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
        try:
            return _paginate(
                snapshot,
                section=section,
                items=items,
                keys=keys,
                limit=limit,
                after=after,
            )
        except ValueError as exc:
            conflict = "another snapshot" in str(exc) or "source window" in str(exc)
            raise HTTPException(
                status_code=409 if conflict else 400, detail=str(exc)
            ) from exc

    @app.get("/api/v1/runtime")
    def runtime(_session: _Session = Depends(session)) -> Mapping[str, Any]:
        return _envelope({"snapshot": snapshot_value()})

    @app.get("/api/v1/profiles")
    def profiles(_session: _Session = Depends(session)) -> Mapping[str, Any]:
        return _envelope(
            {
                "profiles": [
                    {"id": "deepseek/deepseek-v4-pro", "harness": "opencode", "active": True},
                    {"id": "deepseek/deepseek-v4-flash", "harness": "opencode", "active": True},
                ]
            }
        )

    @app.get("/api/v1/tasks")
    def tasks(
        limit: int = 50,
        after: str | None = None,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail="snapshot data unavailable")
        combined: list[Mapping[str, Any]] = [
            {"kind": "run", "value": item} for item in data.get("runs", [])
        ] + [
            {"kind": "campaign", "value": item}
            for item in data.get("campaigns", [])
        ]
        page, page_info = page_value(
            snapshot,
            section="tasks",
            items=combined,
            keys=[f"{item['kind']}:{item['value'].get('id')}" for item in combined],
            limit=limit,
            after=after,
        )
        return _envelope(
            {
                "runs": [item["value"] for item in page if item["kind"] == "run"],
                "campaigns": [
                    item["value"] for item in page if item["kind"] == "campaign"
                ],
                "snapshot_status": snapshot.get("status"),
                "page": page_info,
            }
        )

    @app.get("/api/v1/experiments")
    def experiments(
        limit: int = 50,
        after: str | None = None,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
            raise HTTPException(status_code=503, detail="snapshot section experiments unavailable")
        values = data["experiments"]
        page, page_info = page_value(
            snapshot,
            section="experiments",
            items=values,
            keys=[str(item.get("id")) for item in values],
            limit=limit,
            after=after,
        )
        return _envelope(
            {
                "experiments": page,
                "snapshot_status": snapshot.get("status"),
                "page": page_info,
            }
        )

    @app.get("/api/v1/tasks/{kind}/{task_id}")
    def task_detail(
        kind: str,
        task_id: str,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail="snapshot data unavailable")
        section = "campaigns" if kind in {"campaign", "benchmark"} else "runs"
        selected = next(
            (item for item in data.get(section, []) if str(item.get("id")) == task_id),
            None,
        )
        if selected is None:
            raise HTTPException(status_code=404, detail="task not found")
        related = {
            "iterations": [
                item for item in data.get("iterations", [])
                if str(item.get("run_id")) == task_id
            ],
            "evaluation_attempts": [
                item for item in data.get("evaluation_attempts", [])
                if str(item.get("run_id")) == task_id
            ],
            "child_runs": [
                item for item in data.get("child_runs", [])
                if str(item.get("campaign_id")) == task_id
            ],
        }
        return _envelope({"task": selected, **related})

    @app.get("/api/v1/experiments/{experiment_uid}")
    def experiment_detail(
        experiment_uid: str,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail="snapshot data unavailable")
        selected = next(
            (
                item for item in data.get("experiments", [])
                if item.get("experiment_uid") == experiment_uid
            ),
            None,
        )
        if selected is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        relations = [
            item for item in data.get("experiment_relations", [])
            if experiment_uid in {
                item.get("source_experiment_uid"),
                item.get("target_experiment_uid"),
            }
        ]
        return _envelope({"experiment": selected, "relations": relations})

    @app.get("/api/v1/campaigns")
    def campaigns(
        limit: int = 50,
        after: str | None = None,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("campaigns"), list):
            raise HTTPException(status_code=503, detail="snapshot section campaigns unavailable")
        values = data["campaigns"]
        page, page_info = page_value(
            snapshot,
            section="campaigns",
            items=values,
            keys=[str(item.get("id")) for item in values],
            limit=limit,
            after=after,
        )
        return _envelope(
            {
                "campaigns": page,
                "snapshot_status": snapshot.get("status"),
                "page": page_info,
            }
        )

    @app.get("/api/v1/campaigns/{campaign_id}")
    def campaign_detail(
        campaign_id: str,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail="snapshot data unavailable")
        selected = next(
            (item for item in data.get("campaigns", []) if item.get("id") == campaign_id),
            None,
        )
        if selected is None:
            raise HTTPException(status_code=404, detail="campaign not found")
        return _envelope(
            {
                "campaign": selected,
                "children": [
                    item for item in data.get("child_runs", [])
                    if item.get("campaign_id") == campaign_id
                ],
                "budget_actions": [
                    item for item in data.get("budget_actions", [])
                    if item.get("campaign_id") == campaign_id
                ],
            }
        )

    @app.get("/api/v1/artifacts/{artifact_id:path}")
    def scientific_artifact(
        artifact_id: str,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        try:
            payload = state.transport.call("snapshot", {"artifact_id": artifact_id})
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        artifact = payload.get("scientific_artifact")
        if not isinstance(artifact, dict):
            raise HTTPException(status_code=503, detail="artifact response unavailable")
        return _envelope({"artifact": artifact})

    @app.get("/api/v1/resources")
    def resources(
        limit: int = 50,
        after: str | None = None,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail="snapshot data unavailable")
        combined: list[Mapping[str, Any]] = [
            {"kind": "lease", "value": item}
            for item in data.get("resource_leases", [])
        ] + [
            {"kind": "budget", "value": item}
            for item in data.get("budget_actions", [])
        ]
        page, page_info = page_value(
            snapshot,
            section="resources",
            items=combined,
            keys=[
                (
                    f"lease:{item['value'].get('resource_id')}:"
                    f"{item['value'].get('fencing_epoch')}"
                    if item["kind"] == "lease"
                    else f"budget:{item['value'].get('id')}"
                )
                for item in combined
            ],
            limit=limit,
            after=after,
        )
        return _envelope(
            {
                "resource_leases": [
                    item["value"] for item in page if item["kind"] == "lease"
                ],
                "budget_actions": [
                    item["value"] for item in page if item["kind"] == "budget"
                ],
                "page": page_info,
            }
        )

    @app.get("/api/v1/soak")
    def soak(
        limit: int = 50,
        after: str | None = None,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        snapshot = snapshot_value()
        data = snapshot.get("data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail="snapshot data unavailable")
        combined: list[Mapping[str, Any]] = [
            {"kind": "generation", "value": item}
            for item in data.get("soak_generations", [])
        ] + [
            {"kind": "violation", "value": item}
            for item in data.get("soak_violations", [])
        ]
        page, page_info = page_value(
            snapshot,
            section="soak",
            items=combined,
            keys=[f"{item['kind']}:{item['value'].get('id')}" for item in combined],
            limit=limit,
            after=after,
        )
        return _envelope(
            {
                "soak_generations": [
                    item["value"] for item in page if item["kind"] == "generation"
                ],
                "soak_violations": [
                    item["value"] for item in page if item["kind"] == "violation"
                ],
                "page": page_info,
            }
        )

    @app.get("/api/v1/audit")
    def audit(_session: _Session = Depends(session)) -> Mapping[str, Any]:
        return _envelope({"audit": state.local_store.audit_records()})

    @app.get("/api/v1/drafts")
    def drafts(_session: _Session = Depends(session)) -> Mapping[str, Any]:
        return _envelope({"drafts": state.local_store.list_drafts()})

    @app.put("/api/v1/drafts/{draft_id}")
    async def save_draft(
        draft_id: str,
        request: Request,
        _session: _Session = Depends(mutation_session),
    ) -> Mapping[str, Any]:
        value = await _strict_request_json(request)
        if set(value) != {
            "schema_version",
            "task_kind",
            "title",
            "values",
            "created_at",
            "updated_at",
        } or value.get("schema_version") != 1:
            raise HTTPException(status_code=400, detail="draft contract mismatch")
        try:
            draft = DraftV1(
                draft_id=draft_id,
                task_kind=value["task_kind"],
                title=value["title"],
                values=value["values"],
                created_at=value["created_at"],
                updated_at=value["updated_at"],
            )
            state.local_store.put_draft(draft)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _envelope({"draft": draft.to_dict()})

    @app.get("/api/v1/events/stream")
    async def events(request: Request, _session: _Session = Depends(session)) -> StreamingResponse:
        return StreamingResponse(
            event_broker.stream(request), media_type="text/event-stream"
        )

    @app.post("/api/v1/operations/{kind}/prepare")
    async def prepare_operation(
        kind: str,
        request: Request,
        _session: _Session = Depends(mutation_session),
    ) -> Mapping[str, Any]:
        value = await _strict_request_json(request)
        if set(value) != {
            "schema_version",
            "operation_id",
            "runtime_identity_digest",
            "parameters",
        } or value.get("schema_version") != 1:
            raise HTTPException(status_code=400, detail="prepare contract mismatch")
        snapshot = snapshot_value()
        if snapshot.get("status") != "STABLE":
            raise HTTPException(
                status_code=409, detail="remote snapshot is not stable enough to write"
            )
        runtime = snapshot.get("runtime_identity")
        if (
            not isinstance(runtime, dict)
            or runtime.get("runtime_identity_digest")
            != value["runtime_identity_digest"]
        ):
            raise HTTPException(status_code=409, detail="runtime identity is stale")
        try:
            payload = state.transport.call(
                "prepare",
                {
                    "operation_id": value["operation_id"],
                    "kind": kind,
                    "runtime_identity_digest": value["runtime_identity_digest"],
                    "parameters": value["parameters"],
                },
            )
            prepared = PreparedOperationV1.from_value(
                payload.get("prepared_operation")
            )
            state.local_store.record_prepared(prepared)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _envelope({"prepared_operation": prepared.to_dict()})

    @app.post("/api/v1/operations/{operation_id}/confirm")
    async def confirm_operation(
        operation_id: str,
        request: Request,
        _session: _Session = Depends(mutation_session),
    ) -> Mapping[str, Any]:
        value = await _strict_request_json(request)
        if set(value) != {
            "schema_version",
            "operation_digest",
            "confirmation_phrase",
        } or value.get("schema_version") != 1:
            raise HTTPException(status_code=400, detail="confirm contract mismatch")
        local = state.local_store.get_operation(operation_id)
        if (
            local is None
            or local["status"] != "PREPARED"
            or local["operation_digest"] != value["operation_digest"]
        ):
            raise HTTPException(status_code=409, detail="operation is not prepared")
        try:
            payload = state.transport.call(
                "execute",
                {
                    "operation_id": operation_id,
                    "operation_digest": value["operation_digest"],
                    "confirmation_phrase": value["confirmation_phrase"],
                },
            )
            receipt = OperationReceiptV1.from_value(
                payload.get("operation_receipt")
            )
            state.local_store.record_receipt(receipt)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _envelope({"operation_receipt": receipt.to_dict()})

    @app.get("/api/v1/operations/{operation_id}")
    def operation_status(
        operation_id: str,
        _session: _Session = Depends(session),
    ) -> Mapping[str, Any]:
        local = state.local_store.get_operation(operation_id)
        if local is None:
            raise HTTPException(status_code=404, detail="operation not found")
        if local["status"] in {"PREPARED", "EXECUTING"}:
            try:
                payload = state.transport.call(
                    "reconcile", {"operation_id": operation_id}
                )
                receipt = OperationReceiptV1.from_value(
                    payload.get("operation_receipt")
                )
                state.local_store.record_receipt(receipt)
                local = state.local_store.get_operation(operation_id)
            except (RuntimeError, TypeError, ValueError):
                # A read-side transport outage is not evidence that the remote
                # action failed. Return the last durable mirror without retrying.
                pass
        assert local is not None
        return _envelope({"operation": local})

    assets = state.static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}")
    def spa(path: str, request: Request) -> Response:
        if path.startswith("api/") or "." in Path(path).name:
            raise HTTPException(status_code=404, detail="not found")
        template = (state.static_dir / "index.html").read_text(encoding="utf-8")
        rendered = template.replace("__CSP_NONCE__", request.state.csp_nonce)
        rendered = re.sub(
            r"<(script|link)\b",
            lambda matched: (
                f'<{matched.group(1)} nonce="{request.state.csp_nonce}"'
            ),
            rendered,
        )
        return HTMLResponse(rendered)

    return app


__all__ = [
    "GatewayState",
    "SESSION_COOKIE",
    "SessionRegistry",
    "SnapshotCache",
    "SnapshotEventBroker",
    "create_app",
]
