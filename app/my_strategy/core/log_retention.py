"""Bounded diagnostics for the new dashboard log, separate from audit records."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import logging.handlers
from pathlib import Path
import time

from uvicorn.config import LOGGING_CONFIG

from my_strategy.core.paths import LOG_ROOT


class RetainedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Keep at most three size-bounded backups and expire them after seven days.

    Only this handler's numbered backups are owned here. Other application,
    audit and run logs are deliberately outside its deletion boundary.
    """

    def __init__(self, filename, maxBytes=10 * 1024**2, backupCount=3,
                 retention_days=7, encoding="utf-8", delay=True):
        if maxBytes <= 0 or not 1 <= backupCount <= 3 or retention_days <= 0:
            raise ValueError("positive log bounds and one to three backups required")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = retention_days * 86400
        self._last_expiry_check = 0.0
        super().__init__(path, maxBytes=maxBytes, backupCount=backupCount,
                         encoding=encoding, delay=delay)
        self._rollover_at = self._next_midnight(path.stat().st_mtime if path.is_file() else time.time())
        self._expire_backups()

    def _open(self):
        # Byte accounting below includes LF. Disable platform newline translation
        # so Windows does not add unbudgeted CR bytes, including within messages.
        return self._builtin_open(self.baseFilename, self.mode, encoding=self.encoding,
                                  errors=self.errors, newline="\n")

    @staticmethod
    def _next_midnight(timestamp):
        day = datetime.fromtimestamp(timestamp).replace(hour=0, minute=0, second=0, microsecond=0)
        return (day + timedelta(days=1)).timestamp()

    def _expire_backups(self):
        cutoff = time.time() - self.retention_seconds
        for index in range(1, self.backupCount + 1):
            path = Path(f"{self.baseFilename}.{index}")
            # Never follow a manually introduced alias into an audit log.
            if not path.is_symlink() and path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        self._last_expiry_check = time.time()

    def shouldRollover(self, record):
        path = Path(self.baseFilename)
        if path.is_symlink():
            raise ValueError("managed dashboard log cannot be a symlink")
        # Calendar rotation prevents continuous low-volume writes from keeping
        # arbitrarily old records alive by refreshing the file modification time.
        if path.is_file() and time.time() >= self._rollover_at:
            return True
        if path.is_file() and path.stat().st_mtime < time.time() - self.retention_seconds:
            return True
        return path.is_file() and path.stat().st_size + len((self.format(record) + "\n").encode(self.encoding)) >= self.maxBytes

    def format(self, record):
        text = super().format(record)
        encoded = text.encode(self.encoding)
        if len(encoded) >= self.maxBytes:
            suffix = b" [truncated]"
            text = encoded[:max(0, self.maxBytes - len(suffix) - 1)].decode(self.encoding, errors="ignore")
            text += suffix.decode() if self.maxBytes > len(suffix) else ""
        return text

    def doRollover(self):
        super().doRollover()
        self._rollover_at = self._next_midnight(time.time())
        self._expire_backups()

    def emit(self, record):
        if time.time() - self._last_expiry_check >= 3600:
            self._expire_backups()
        super().emit(record)


def dashboard_logging_config(log_root=None):
    config = deepcopy(LOGGING_CONFIG)
    config["formatters"]["retained"] = {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
    config["handlers"]["retained"] = {
        "class": "my_strategy.core.log_retention.RetainedRotatingFileHandler",
        "filename": str(Path(log_root or LOG_ROOT) / "dashboard" / "web.log"),
        "formatter": "retained", "maxBytes": 10 * 1024**2,
        "backupCount": 3, "retention_days": 7,
    }
    config["loggers"]["uvicorn"]["handlers"].append("retained")
    config["loggers"]["uvicorn.access"]["handlers"].append("retained")
    config["root"] = {"level": "INFO", "handlers": ["retained"]}
    return config
