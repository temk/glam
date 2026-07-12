import pytest

from glam.common.hooks import HookError, service_hooks
from glam.common.config import HookConfig, ServiceHooks


class _FakeResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def urlopen(monkeypatch):
    """Record hook requests and drive their outcome; patches urllib.request.urlopen."""
    calls: list[dict] = []

    def install(status: int = 200, error: Exception | None = None):
        def fake(request, timeout):
            calls.append({"url": request.full_url, "method": request.get_method(), "timeout": timeout})
            if error is not None:
                raise error
            return _FakeResponse(status)

        monkeypatch.setattr("urllib.request.urlopen", fake)
        return calls

    return install


def _hooks(pre: bool = False, post: bool = False) -> ServiceHooks:
    return ServiceHooks(
        pre=HookConfig(url="http://h/pre", timeout=120) if pre else None,
        post=HookConfig(url="http://h/post", timeout=60) if post else None,
    )


def test_pre_runs_before_work_and_post_after(urlopen):
    calls = urlopen()
    order: list[str] = []
    with service_hooks(_hooks(pre=True, post=True), echo=lambda *_: None):
        order.append("work")

    assert [c["url"] for c in calls] == ["http://h/pre", "http://h/post"]
    assert calls[0]["method"] == "POST" and calls[0]["timeout"] == 120
    assert order == ["work"]


def test_post_runs_even_when_work_fails(urlopen):
    calls = urlopen()
    with pytest.raises(ValueError):
        with service_hooks(_hooks(post=True), echo=lambda *_: None):
            raise ValueError("boom")

    assert [c["url"] for c in calls] == ["http://h/post"]  # post still fired in finally


def test_pre_failure_cancels_work_and_skips_post(urlopen):
    calls = urlopen(status=500)
    ran = False
    with pytest.raises(HookError):
        with service_hooks(_hooks(pre=True, post=True), echo=lambda *_: None):
            ran = True

    assert ran is False  # work never started
    assert [c["url"] for c in calls] == ["http://h/pre"]  # post was not called


def test_post_failure_fails_a_successful_step(urlopen):
    urlopen(status=503)
    with pytest.raises(HookError):
        with service_hooks(_hooks(post=True), echo=lambda *_: None):
            pass  # step itself succeeds


def test_post_failure_does_not_mask_step_error(urlopen):
    urlopen(status=503)  # post hook would fail...
    with pytest.raises(ValueError):  # ...but the original step error stays primary
        with service_hooks(_hooks(post=True), echo=lambda *_: None):
            raise ValueError("step failed first")


def test_no_hooks_is_a_noop(urlopen):
    calls = urlopen()
    with service_hooks(None, echo=lambda *_: None):
        pass

    assert calls == []
