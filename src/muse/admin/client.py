"""AdminClient: thin Python wrapper over the /v1/admin/* HTTP surface.

Use this for programmatic admin against a running `muse serve`. For
in-process usage (no HTTP), import the operations module directly.

Token resolution:
  1. constructor `token=` arg
  2. MUSE_ADMIN_TOKEN env var
  3. None (every call will 503 since the server requires the env var)

Server resolution:
  1. constructor `base_url=` arg
  2. MUSE_SERVER env var
  3. http://localhost:8000

The `wait` helper polls `/jobs/{id}` until the job lands in done/failed.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx


class AdminClientError(Exception):
    """Raised when an admin call returns a non-2xx response.

    `code` is the OpenAI error envelope's `code` field; `status` is
    the HTTP status. `body` is the raw decoded JSON.
    """

    def __init__(self, status: int, code: str, message: str, body: Any):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.body = body


class AdminClient:
    """HTTP client for /v1/admin/* admin endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (
            base_url
            or os.environ.get("MUSE_SERVER")
            or "http://localhost:8000"
        ).rstrip("/")
        self.token = token or os.environ.get("MUSE_ADMIN_TOKEN")
        self._timeout = timeout

    def _headers(self) -> dict:
        if self.token is None:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.request(
                method,
                url,
                headers={**self._headers(), **kwargs.pop("headers", {})},
                **kwargs,
            )
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:  # noqa: BLE001
                body = {"raw": r.text}
            err = body.get("error") or body.get("detail", {}).get("error") or {}
            code = err.get("code", "http_error")
            message = err.get("message", r.text)
            raise AdminClientError(r.status_code, code, message, body)
        try:
            return r.json()
        except Exception:  # noqa: BLE001
            return {"raw": r.text}

    # Per-model operations

    def enable(self, model_id: str) -> dict:
        return self._request("POST", f"/v1/admin/models/{model_id}/enable", json={})

    def disable(self, model_id: str) -> dict:
        return self._request("POST", f"/v1/admin/models/{model_id}/disable", json={})

    def probe(
        self,
        model_id: str,
        *,
        no_inference: bool = False,
        device: str | None = None,
    ) -> dict:
        body = {"no_inference": no_inference}
        if device is not None:
            body["device"] = device
        return self._request("POST", f"/v1/admin/models/{model_id}/probe", json=body)

    def pull(self, identifier: str) -> dict:
        # Use the documented `_` placeholder path; identifier in body
        # avoids URL-encoding hf://... slashes.
        return self._request(
            "POST",
            "/v1/admin/models/_/pull",
            json={"identifier": identifier},
        )

    def remove(self, model_id: str, *, purge: bool = False) -> dict:
        return self._request(
            "DELETE",
            f"/v1/admin/models/{model_id}",
            params={"purge": "true" if purge else "false"},
        )

    def status(self, model_id: str) -> dict:
        return self._request("GET", f"/v1/admin/models/{model_id}/status")

    # Cluster-wide views

    def memory(self) -> dict:
        return self._request("GET", "/v1/admin/memory")

    def workers(self) -> dict:
        return self._request("GET", "/v1/admin/workers")

    def restart_worker(self, port: int) -> dict:
        return self._request("POST", f"/v1/admin/workers/{port}/restart")

    # Job tracking

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/admin/jobs/{job_id}")

    def jobs(self) -> dict:
        return self._request("GET", "/v1/admin/jobs")

    def wait(
        self,
        job_id: str,
        *,
        timeout: float = 300.0,
        poll: float = 1.0,
    ) -> dict:
        """Block until job is done or failed; return the final job record.

        Raises TimeoutError if the job never reaches a terminal state
        within `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self.job(job_id)
            if job.get("state") in ("done", "failed"):
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id} did not finish within {timeout}s "
                    f"(last state: {job.get('state')})"
                )
            time.sleep(poll)
