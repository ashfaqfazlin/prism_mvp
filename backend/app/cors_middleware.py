"""Permissive CORS at the ASGI layer (OPTIONS short-circuit + headers on all responses)."""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _header_names(headers: list[tuple[bytes, bytes]]) -> set[bytes]:
    return {h[0].lower() for h in headers}


class AllowAnyOriginCorsMiddleware:
    """Allow any browser origin; short-circuit OPTIONS so preflight never hits the router."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["method"] == "OPTIONS":
            req = Headers(scope=scope)
            allow_headers = req.get("access-control-request-headers") or "*"
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (b"access-control-allow-origin", b"*"),
                        (
                            b"access-control-allow-methods",
                            b"DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
                        ),
                        (b"access-control-allow-headers", allow_headers.encode("latin-1")),
                        (b"access-control-max-age", b"86400"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers") or [])
                if b"access-control-allow-origin" not in _header_names(hdrs):
                    hdrs.append((b"access-control-allow-origin", b"*"))
                message = {**message, "headers": hdrs}
            await send(message)

        await self.app(scope, receive, send_with_cors)
