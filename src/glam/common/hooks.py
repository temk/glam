import urllib.error
import urllib.request
from typing import Iterator
from contextlib import contextmanager

from glam.common.config import HookConfig, ServiceHooks
from glam.common.errors import GlamError


class HookError(GlamError):
    pass


@contextmanager
def service_hooks(hooks: ServiceHooks | None, echo) -> Iterator[None]:
    """Run a service's `pre` hook before the wrapped work and its `post` hook after, once per step.

    `pre` runs before the work; its failure propagates so the step is never started (and `post` is
    then skipped). `post` runs in `finally`. A `post` failure is a step failure — it propagates when
    the work itself succeeded; when the work already raised, that original error stays primary and the
    `post` failure is only logged, so a failure in `finally` never masks the real cause.
    """
    if hooks is not None and hooks.pre is not None:
        _call("pre", hooks.pre, echo)
    step_ok = False
    try:
        yield
        step_ok = True
    finally:
        if hooks is not None and hooks.post is not None:
            try:
                _call("post", hooks.post, echo)
            except HookError:
                if step_ok:
                    raise
                echo("post hook failed too; keeping the original step error")


def _call(kind: str, hook: HookConfig, echo) -> None:
    echo(f"{kind} hook: {hook.method} {hook.url}")
    request = urllib.request.Request(hook.url, method=hook.method)
    try:
        with urllib.request.urlopen(request, timeout=hook.timeout) as response:
            status = response.status
    except urllib.error.HTTPError as e:
        raise HookError(f"{kind} hook {hook.url} returned HTTP {e.code}") from e
    except (urllib.error.URLError, OSError) as e:
        raise HookError(f"{kind} hook {hook.url} failed: {e}") from e
    if not 200 <= status < 300:
        raise HookError(f"{kind} hook {hook.url} returned HTTP {status}")
