"""Monkey-patch prometheus_fastapi_instrumentator routing bug at import time.

The installed version crashes on _IncludedRouter objects that lack a .path
attribute. This installs a patched _get_route_name into the module so all
subsequent imports see the fix.

Must be run before vLLM starts (i.e., early in the entrypoint).
"""
import sys

try:
    import prometheus_fastapi_instrumentator.routing as routing_mod
except ImportError:
    print("prometheus_fastapi_instrumentator not installed, skipping patch")
    sys.exit(0)

original = getattr(routing_mod, "_get_route_name", None)
if original is None:
    print("_get_route_name not found, skipping patch")
    sys.exit(0)

def _get_route_name_patched(scope, routes):
    for route in routes:
        if hasattr(route, "path"):
            match, _ = route.matches(scope)
            if match.name == "FULL":
                return route.path
        elif hasattr(route, "routes"):
            result = _get_route_name_patched(scope, route.routes)
            if result:
                return result
    return None

routing_mod._get_route_name = _get_route_name_patched
print("Patched prometheus_fastapi_instrumentator.routing._get_route_name (monkey-patch)")
