"""Order-failure classifier — buckets every failed real trade into one of 12 modes.

Used by user_real_manager._execute_shadow / _execute_real to tag the failure
mode on a parity_audit row so we can aggregate by reason.

The 12 buckets (architect-defined 2026-05-03):
  API_REJECTED         — Delta returned 4xx/5xx but not specifically auth/margin/etc
  AUTH_ERROR           — 401/403 / "invalid_api_key" / "signature mismatch"
  PRODUCT_ID_MISMATCH  — wrong contract / unknown symbol id
  SIZE_TOO_SMALL       — order qty below minimum lot
  INSUFFICIENT_MARGIN  — balance check fail / "insufficient_margin"
  PRICE_OUT_OF_BOUNDS  — limit too far from mark / circuit-breaker / "price_band"
  POST_ONLY_REJECTED   — maker order would have been taker (post-only flag rejected)
  ORDER_TIMEOUT        — network/HTTP timeout, no response
  PARTIAL_FILL         — filled qty < requested qty (still recorded as success but flagged)
  NO_FILL              — limit order didn't fill in the time window
  SL_TP_NOT_PLACED     — entry filled but bracket (SL or TP) order placement failed
  MONITOR_MISSED_EXIT  — price hit TP/SL on tape but our monitor loop didn't trigger close

Classifier accepts:
  - an Exception (network / library)
  - a response dict (Delta API JSON body)
  - a free-form context tag for non-API failures (NO_FILL, MONITOR_MISSED_EXIT)

Returns a (bucket, detail) tuple. detail is a short human-readable string.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

BUCKETS = (
    "API_REJECTED",
    "AUTH_ERROR",
    "PRODUCT_ID_MISMATCH",
    "SIZE_TOO_SMALL",
    "INSUFFICIENT_MARGIN",
    "PRICE_OUT_OF_BOUNDS",
    "POST_ONLY_REJECTED",
    "ORDER_TIMEOUT",
    "PARTIAL_FILL",
    "NO_FILL",
    "SL_TP_NOT_PLACED",
    "MONITOR_MISSED_EXIT",
)


# Patterns for matching against error strings (lowercase comparison)
_PATTERNS = {
    "AUTH_ERROR": [
        r"unauthor", r"\binvalid api", r"signature.*mismatch", r"signature.*invalid",
        r"\bapi[\-_ ]?key", r"\bauth(entication)?\b.*(fail|error)", r"\bforbidden\b",
        r"http\s*40[13]\b",
    ],
    "PRODUCT_ID_MISMATCH": [
        r"product[_\s]*id.*(invalid|mismatch|unknown|not.*found)",
        r"unknown[_\s]*symbol", r"contract.*not[_\s]*found",
        r"invalid[_\s]*product",
    ],
    "SIZE_TOO_SMALL": [
        r"\bsize.*(too[_\s]*small|min[i]?mum|below)", r"\bqty.*(too[_\s]*small|min)",
        r"\bminimum[_\s]*(order|size|lot|qty)", r"\bmin[_\s]*notional",
        r"\bquantity.*(too[_\s]*small|below[_\s]*minimum)",
    ],
    "INSUFFICIENT_MARGIN": [
        r"insufficient[_\s]*(margin|balance|funds)", r"\bnot[_\s]*enough[_\s]*(margin|balance|funds)",
        r"\bmargin[_\s]*(call|short|insufficient)", r"\bbalance[_\s]*(too[_\s]*low|insufficient)",
    ],
    "PRICE_OUT_OF_BOUNDS": [
        r"price[_\s]*(band|out[_\s]*of|too[_\s]*far|deviate)", r"price[_\s]*deviation",
        r"limit.*(too[_\s]*far|outside|out[_\s]*of[_\s]*range)",
        r"circuit[_\s]*breaker", r"\btrigger[_\s]*price.*(invalid|range)",
    ],
    "POST_ONLY_REJECTED": [
        r"post[_\s\-]?only.*(reject|cross|crossing|would[_\s]*have)",
        r"would[_\s]*cross[_\s]*(spread|book)",
        r"\bmaker[_\s]*only.*(reject|fail)", r"crossed[_\s]*spread",
    ],
    "ORDER_TIMEOUT": [
        r"\btimeout\b", r"timed[_\s]*out", r"connection[_\s]*(reset|aborted|refused)",
        r"network[_\s]*error", r"\bgateway[_\s]*timeout\b", r"http\s*504\b",
    ],
}

# Compile once
_COMPILED: Dict[str, list] = {
    bucket: [re.compile(p, re.IGNORECASE) for p in pats]
    for bucket, pats in _PATTERNS.items()
}


def _scan_text(text: str) -> Optional[str]:
    """Return matching bucket for a free-form error string, or None."""
    if not text:
        return None
    # Order matters: AUTH first (4xx code), then specific reasons, then generic timeout
    for bucket in (
        "AUTH_ERROR", "PRODUCT_ID_MISMATCH", "SIZE_TOO_SMALL",
        "INSUFFICIENT_MARGIN", "PRICE_OUT_OF_BOUNDS",
        "POST_ONLY_REJECTED", "ORDER_TIMEOUT",
    ):
        for pat in _COMPILED[bucket]:
            if pat.search(text):
                return bucket
    return None


def classify(
    *,
    exception: Optional[BaseException] = None,
    response: Optional[Dict[str, Any]] = None,
    context_tag: Optional[str] = None,
    requested_qty: Optional[float] = None,
    filled_qty: Optional[float] = None,
) -> Tuple[str, str]:
    """Classify an order-placement failure or post-fill anomaly.

    Args:
        exception   — raised by the API client (network, http, etc.)
        response    — Delta API response body (dict). May contain 'success', 'error'.
        context_tag — explicit tag for non-API states ("NO_FILL_TIMEOUT",
                       "MONITOR_NO_TRIGGER", "BRACKET_FAIL", "PARTIAL_FILL").
        requested_qty + filled_qty — for PARTIAL_FILL detection.

    Returns:
        (bucket, detail) where bucket ∈ BUCKETS and detail is a short string.

    Always returns a value — never raises. Falls through to API_REJECTED if
    no specific match.
    """
    # ── Explicit context tags (highest precedence — they describe non-API states) ──
    if context_tag:
        ct = context_tag.upper()
        if ct in ("NO_FILL", "NO_FILL_TIMEOUT", "MAKER_TIMEOUT"):
            return "NO_FILL", context_tag
        if ct in ("MONITOR_MISSED_EXIT", "MONITOR_NO_TRIGGER", "EXIT_NOT_TRIGGERED"):
            return "MONITOR_MISSED_EXIT", context_tag
        if ct in ("BRACKET_FAIL", "SL_NOT_PLACED", "TP_NOT_PLACED", "SL_TP_NOT_PLACED"):
            return "SL_TP_NOT_PLACED", context_tag

    # ── Partial-fill detection ──
    if (requested_qty is not None and filled_qty is not None
        and filled_qty > 0 and filled_qty < requested_qty - 1e-9):
        return "PARTIAL_FILL", f"filled={filled_qty}/{requested_qty}"

    # ── Exception path ──
    if exception is not None:
        msg = str(exception)
        # Check exception class name first
        cls = type(exception).__name__.lower()
        if "timeout" in cls or "timeouterror" in cls:
            return "ORDER_TIMEOUT", f"{type(exception).__name__}: {msg[:120]}"
        if "connection" in cls:
            return "ORDER_TIMEOUT", f"{type(exception).__name__}: {msg[:120]}"
        # Then scan the message
        b = _scan_text(msg)
        if b is not None:
            return b, msg[:200]
        return "API_REJECTED", f"{type(exception).__name__}: {msg[:200]}"

    # ── Response dict path ──
    if response is not None:
        # Delta-style: {"success": false, "error": {"code": "...", "message": "..."}}
        success = response.get("success", True)
        if success:
            # Check for partial-fill flags
            req = response.get("size") or response.get("qty") or response.get("requested_qty")
            fil = response.get("filled_size") or response.get("executed_qty") or response.get("filled_qty")
            if (req is not None and fil is not None
                and float(fil) > 0 and float(fil) < float(req) - 1e-9):
                return "PARTIAL_FILL", f"filled={fil}/{req}"
            # No failure detected
            return "API_REJECTED", "response.success=true but classify() called — caller bug?"
        # success = false
        err = response.get("error", {})
        if isinstance(err, str):
            err_text = err
            err_code = ""
        else:
            err_text = str(err.get("message", ""))
            err_code = str(err.get("code", ""))
        full = f"{err_code} {err_text}".strip()
        b = _scan_text(full)
        if b is not None:
            return b, full[:200]
        return "API_REJECTED", full[:200] or "unspecified API error"

    # ── No info at all ──
    return "API_REJECTED", "classify() called with no exception/response/context"


def is_retryable(bucket: str) -> bool:
    """Hint for retry-loop authors. Not used by classifier itself."""
    return bucket in {"ORDER_TIMEOUT", "PRICE_OUT_OF_BOUNDS"}


def is_terminal_account_issue(bucket: str) -> bool:
    """If True — pause the user's trading; don't keep hammering the API."""
    return bucket in {"AUTH_ERROR", "INSUFFICIENT_MARGIN", "PRODUCT_ID_MISMATCH"}
