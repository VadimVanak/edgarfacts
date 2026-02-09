# src/edgarfacts/fetching.py
"""
HTTP fetching utilities for edgarfacts.

This module provides a small, SEC-friendly URL fetcher:
- Adds SEC-required User-Agent for sec.gov / data.sec.gov endpoints
- Enforces a minimum delay between SEC requests (default 0.11s)
- Supports "ignore_exceptions" mode for best-effort retrieval
"""

from __future__ import annotations

import os
import ssl
import time
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen

import logging


@dataclass(frozen=True)
class FetcherConfig:
    """
    Configuration for URLFetcher.

    Parameters
    ----------
    user_agent:
        User-Agent string required by SEC for requests to sec.gov / data.sec.gov.
        If None, URLFetcher tries environment variables and notebook secrets.
    min_interval_sec:
        Minimum time interval between SEC requests (rate limiting).
    ssl_context:
        Optional SSL context. If None, uses default verified context.
    """

    user_agent: Optional[str] = None
    min_interval_sec: float = 0.11
    ssl_context: Optional[ssl.SSLContext] = None


class URLFetcher:
    """
    A small HTTP fetcher with SEC EDGAR-specific behavior.

    Notes
    -----
    SEC asks automated tools to include a descriptive User-Agent and to respect
    fair access / rate limits. This class enforces a minimum inter-request delay
    for SEC domains.
    """

    _SEC_PREFIXES = ("https://www.sec.gov", "https://data.sec.gov")

    def __init__(self, logger: logging.Logger, config: Optional[FetcherConfig] = None):
        self._logger = logger
        self._config = config or FetcherConfig()
        self._user_agent = self._resolve_user_agent(self._config.user_agent)

        # Ensure we don't immediately sleep on the first call.
        self._prev_req_time = time.time() - float(self._config.min_interval_sec)

        # SSL context: default verified unless explicitly overridden.
        self._ssl_context = self._config.ssl_context

    def _resolve_user_agent(self, user_agent: Optional[str]) -> str:
        """
        Resolve the SEC User-Agent.

        Resolution order:
        1) Explicit parameter
        2) Environment variables
        3) Kaggle/Colab secrets (best effort)
        4) Fallback placeholder (still a string, but you should override in production)
        """
        if user_agent and user_agent.strip():
            return user_agent.strip()

        # Common environment variables
        for key in ("EDGAR_USER_AGENT", "SEC_USER_AGENT", "EDGAR_USERNAME", "EDGARFACTS_USER_AGENT"):
            val = os.getenv(key)
            if val and val.strip():
                return val.strip()

        # Best-effort: Kaggle secret
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            val = UserSecretsClient().get_secret("edgar_username")
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            pass

        # Best-effort: Colab userdata
        try:
            from google.colab import userdata  # type: ignore

            val = userdata.get("edgar_username")
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            pass

        # Last resort: still return something to avoid None headers,
        # but users should override this in production.
        return "edgarfacts (missing User-Agent; set EDGAR_USER_AGENT)"

    def _limit_request_ratio(self) -> None:
        """
        Enforce a minimum time interval between consecutive SEC requests.
        """
        min_interval = float(self._config.min_interval_sec)
        sleep_for = max(min_interval - (time.time() - self._prev_req_time), 0.001)
        time.sleep(sleep_for)
        self._prev_req_time = time.time()

    def fetch(self, url: str, ignore_exceptions: bool = False):
        """
        Fetch the content from the specified URL.

        Parameters
        ----------
        url:
            URL to fetch.
        ignore_exceptions:
            If True, return None on errors and log a warning.
            If False, propagate exceptions.

        Returns
        -------
        http.client.HTTPResponse | None
            Response object (context-manageable) or None in ignore_exceptions mode.
        """
        req = Request(url)

        if url.startswith(self._SEC_PREFIXES):
            req.add_header("User-Agent", self._user_agent)
            self._limit_request_ratio()

        if not ignore_exceptions:
            if self._ssl_context is not None:
                return urlopen(req, context=self._ssl_context)
            return urlopen(req)

        try:
            if self._ssl_context is not None:
                return urlopen(req, context=self._ssl_context)
            return urlopen(req)
        except Exception as e:
            self._logger.warning(f"Failed to fetch {url}: {e}")
            return None
