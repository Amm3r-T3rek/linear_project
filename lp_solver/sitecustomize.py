"""Compatibility shim for running Streamlit behind the Caddy gateway.

The gateway forwards WebSocket upgrade requests to Streamlit's static-file
routes, which trips ``assert scope["type"] == "http"`` inside Starlette's
``StaticFiles`` and crashes the worker.  This shim makes ``StaticFiles``
reject WebSocket scopes with a clean 403 instead of raising, so the server
stays alive.  It is a no-op when Streamlit is run normally (``streamlit run
app.py``) because no WebSocket ever reaches a static route in that case.
"""

try:
    from starlette.staticfiles import StaticFiles
    from starlette.responses import Response

    _orig_call = StaticFiles.__call__

    async def _safe_call(self, scope, receive, send):
        if scope.get("type") == "websocket":
            # Politely reject the WebSocket so Starlette doesn't assert.
            await send({"type": "websocket.close", "code": 1003})
            return
        return await _orig_call(self, scope, receive, send)

    StaticFiles.__call__ = _safe_call
except Exception:
    # Starlette not importable yet (or already patched) — ignore.
    pass
