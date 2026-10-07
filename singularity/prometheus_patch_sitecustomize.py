"""sitecustomize hook: patch prometheus_fastapi_instrumentator on every Python startup.

Place this where PYTHONPATH points so it runs automatically in every subprocess
(including vLLM server actors spawned by Ray).
"""
try:
    import prometheus_fastapi_instrumentator.routing as _r

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

    _r._get_route_name = _get_route_name_patched
except Exception:
    pass
