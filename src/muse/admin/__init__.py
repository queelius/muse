"""Admin REST API for runtime model control.

Mounted on the gateway under /v1/admin/*; gated by MUSE_ADMIN_TOKEN.
See docs/superpowers/specs/2026-04-28-admin-api-design.md for the full
wire contract; this package provides:

  - auth: bearer-token verification dependency for FastAPI
  - jobs: in-memory async-job tracker with 10-minute retention
  - operations: orchestrates enable / disable / probe / pull / remove
                via the supervisor singleton
  - routes/: per-resource APIRouter modules
  - client: thin Python wrapper for programmatic admin access

The admin surface is closed-by-default. Without MUSE_ADMIN_TOKEN set,
all admin requests return 503 admin_disabled. With the env var set,
the request must carry Authorization: Bearer <token>.
"""
