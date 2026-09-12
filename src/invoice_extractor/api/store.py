"""
Temporary in-memory results store.

This exists ONLY because Postgres isn't wired up until Phase 5, and
GET /stats needs something to compute aggregates from right now. This
entire module gets deleted in Phase 5 and replaced with real Postgres
queries — nothing here is meant to survive past that point, so keep it
as small as possible rather than building it out further.
"""

from dataclasses import dataclass, field


@dataclass
class RequestLog:
    """One record of a single /extract call, kept only for this process's lifetime."""

    success: bool
    latency_seconds: float
    estimated_cost_usd: float


@dataclass
class InMemoryStore:
    """Not thread-safe, not persistent, resets on every server restart —
    that's expected and fine for a Phase 3 placeholder."""

    logs: list[RequestLog] = field(default_factory=list)
    _logged_job_ids: set = field(default_factory=set)  
    
    def add(self, log: RequestLog) -> None:
        self.logs.append(log)
        
    def add_once(self, job_id: str, log: RequestLog) -> None:
        """
        Guards against double-counting: a client polling GET /jobs/{id}
        in a loop will hit the 'complete' branch repeatedly after the
        job finishes. Without this, /stats would inflate every time
        someone re-polls an already-finished job.
        """
        if job_id in self._logged_job_ids:
            return
        self._logged_job_ids.add(job_id)
        self.logs.append(log)
        
    def compute_stats(self) -> dict:
        if not self.logs:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "success_rate": 0.0,
                "average_latency_seconds": None,
                "total_estimated_cost_usd": 0.0,
            }

        total = len(self.logs)
        successful = sum(1 for log in self.logs if log.success)

        return {
            "total_requests": total,
            "successful_requests": successful,
            "success_rate": round(successful / total, 4),
            "average_latency_seconds": round(sum(l.latency_seconds for l in self.logs) / total, 3),
            "total_estimated_cost_usd": round(sum(l.estimated_cost_usd for l in self.logs), 6),
        }


# Single shared instance for the app's lifetime — module-level singleton,
# same pattern as `settings` in config.py.
store = InMemoryStore()