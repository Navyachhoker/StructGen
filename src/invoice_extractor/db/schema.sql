-- One record per completed extraction attempt.
--
-- model_name is included now (not added later) because the project's
-- core purpose is comparing multiple model paths (Phase 7 adds a
-- second one) — adding this column after data already exists would
-- mean a migration + backfill instead of just filling it in going forward.
CREATE TABLE IF NOT EXISTS extraction_requests (
    id              BIGSERIAL PRIMARY KEY,
    job_id          TEXT NOT NULL UNIQUE,
    model_name      TEXT NOT NULL,
    success         BOOLEAN NOT NULL,
    invoice_json    JSONB,              -- NULL when extraction failed
    error           TEXT,               -- NULL when extraction succeeded
    latency_seconds DOUBLE PRECISION NOT NULL,
    estimated_cost_usd NUMERIC(12, 6) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Speeds up per-model breakdowns in /stats once Phase 7 adds a second model.
CREATE INDEX IF NOT EXISTS idx_extraction_requests_model_name
    ON extraction_requests (model_name);