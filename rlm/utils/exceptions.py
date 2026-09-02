"""Custom exceptions for RLM execution limits and cancellation."""


class BudgetExceededError(Exception):
    """Raised when the RLM execution exceeds the maximum budget."""

    def __init__(self, spent: float, budget: float, message: str | None = None):
        self.spent = spent
        self.budget = budget
        super().__init__(message or f"Budget exceeded: spent ${spent:.6f} of ${budget:.6f} budget")


class TimeoutExceededError(Exception):
    """Raised when the RLM execution exceeds the maximum timeout."""

    def __init__(
        self,
        elapsed: float,
        timeout: float,
        partial_answer: str | None = None,
        message: str | None = None,
    ):
        self.elapsed = elapsed
        self.timeout = timeout
        self.partial_answer = partial_answer
        super().__init__(message or f"Timeout exceeded: {elapsed:.1f}s of {timeout:.1f}s limit")


class TokenLimitExceededError(Exception):
    """Raised when the RLM execution exceeds the maximum token limit."""

    def __init__(
        self,
        tokens_used: int,
        token_limit: int,
        partial_answer: str | None = None,
        message: str | None = None,
    ):
        self.tokens_used = tokens_used
        self.token_limit = token_limit
        self.partial_answer = partial_answer
        super().__init__(
            message or f"Token limit exceeded: {tokens_used:,} of {token_limit:,} tokens"
        )


class ErrorThresholdExceededError(Exception):
    """Raised when the RLM encounters too many consecutive errors."""

    def __init__(
        self,
        error_count: int,
        threshold: int,
        last_error: str | None = None,
        partial_answer: str | None = None,
        message: str | None = None,
    ):
        self.error_count = error_count
        self.threshold = threshold
        self.last_error = last_error
        self.partial_answer = partial_answer
        super().__init__(
            message
            or f"Error threshold exceeded: {error_count} consecutive errors (limit: {threshold})"
        )


class CancellationError(Exception):
    """Raised when the RLM execution is cancelled by the user."""

    def __init__(self, partial_answer: str | None = None, message: str | None = None):
        self.partial_answer = partial_answer
        super().__init__(message or "Execution cancelled by user")


# better-rlm: added by this fork. It sits beside the engine's own limit exceptions rather
# than under the marker banner below because core/rlm.py catches it in the same tuple.
class SessionBudgetError(Exception):
    """Raised when the backend refuses a call because the caller's SESSION budget --
    a rolling usage window, not a per-run dollar cap -- would be crossed.

    Distinct from BudgetExceededError (this run's own $ budget) because it is a
    property of the account's window, shared with everything else the operator is
    running, and because it is the one limit a caller can RESUME from: the loop
    attaches its transcript and REPL state so the next call continues rather than
    restarts. Carries partial_answer like the other limits.
    """

    def __init__(self, partial_answer: str | None = None, message: str | None = None):
        self.partial_answer = partial_answer
        super().__init__(message or "Session budget stop: the next call would cross the stop line")


# --- better-rlm --------------------------------------------------------------
#: Attribute a backend sets on an exception to say "this failure is a property of
#: the session, not of this prompt" - an expired login, an exhausted quota. Every
#: batched fan-out in the engine stops on it instead of reissuing the identical
#: doomed call once per prompt.
#:
#: Lives here, not in environments/base_env.py, because core/lm_handler.py fans out
#: too and core/ must not import from environments/. This module imports nothing
#: from rlm, so both layers can reach it.
FATAL_SUBCALL_ATTR = "is_fatal_subcall"


def aborts_batch(exc: BaseException) -> bool:
    """True when one sub-call failure condemns the whole batch.

    Duck-typed on purpose: the engine never imports the backend that raised, it
    only honours the documented attribute. See FATAL_SUBCALL_ATTR.
    """
    return bool(getattr(exc, FATAL_SUBCALL_ATTR, False))


#: Attribute a backend sets on an exception to say "stop the WHOLE run cleanly" -- the
#: caller's session window is about to be exhausted. Sibling of FATAL_SUBCALL_ATTR
#: (which only aborts a fan-out): the root loop converts an exception carrying this into
#: SessionBudgetError with the best partial answer and a resumable checkpoint attached,
#: instead of letting it escape as a traceback.
STOPS_RUN_ATTR = "is_session_budget_stop"


def stops_run(exc: BaseException) -> bool:
    """True when a backend failure means the run itself must stop, resumably."""
    return bool(getattr(exc, STOPS_RUN_ATTR, False))
