"""HuggingFace 下载进度工具：nohup / 重定向到文件时也能看到进度。"""

from __future__ import annotations

import sys
import time
from typing import Any

from tqdm import tqdm as StdTqdm


def setup_line_buffered_stdout() -> None:
    """让 print / tqdm 在重定向到日志文件时尽快落盘。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True, write_through=True)
            except (ValueError, OSError):
                pass


def log_print(*args: Any, sep: str = " ", end: str = "\n", **kwargs: Any) -> None:
    """始终 flush 的 print。"""
    print(*args, sep=sep, end=end, flush=True, **kwargs)


def _format_bytes(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class LogFileTqdm(StdTqdm):
    """非 TTY（nohup / 日志文件）下按行输出进度，避免 \\r 刷新不可见。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("name", None)
        self._log_interval = float(kwargs.pop("log_interval", 15.0))
        self._line_log = not sys.stdout.isatty()
        self._last_log_time = 0.0
        self._last_log_pct = -1
        self._desc_label = kwargs.get("desc", "进度")
        self._closed_logged = False

        kwargs.setdefault("file", sys.stdout)
        if self._line_log:
            # 非 TTY：禁止 tqdm 用 \\r 刷新，改由 _maybe_log_progress 按行输出
            kwargs["disable"] = False
            kwargs["dynamic_ncols"] = False
            kwargs.setdefault("mininterval", self._log_interval)
        else:
            kwargs.setdefault("dynamic_ncols", True)

        super().__init__(*args, **kwargs)

    def refresh(self, nolock: bool = False, lock_args=None) -> None:
        if self._line_log:
            return
        super().refresh(nolock=nolock, lock_args=lock_args)

    def update(self, n: float = 1) -> bool | None:
        result = super().update(n)
        if self._line_log:
            self._maybe_log_progress(force=False)
        return result

    def close(self) -> None:
        if self._line_log and not self._closed_logged:
            self._closed_logged = True
            desc = getattr(self, "desc", None) or self._desc_label
            log_print(f"{desc}: 完成 ({_format_bytes(self.n)})")
            self.disable = True
        super().close()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        if desc:
            self._desc_label = desc
        super().set_description(desc, refresh=refresh)
        if self._line_log and desc:
            log_print(f"{desc}")

    def _maybe_log_progress(self, *, force: bool) -> None:
        now = time.time()
        if not force and now - self._last_log_time < self._log_interval:
            return

        desc = getattr(self, "desc", None) or self._desc_label
        if self.total and self.total > 0:
            pct = min(100, int(100 * self.n / self.total))
            if not force and pct <= self._last_log_pct:
                return
            self._last_log_pct = pct
            rate = self.format_dict.get("rate")
            rate_str = f", {_format_bytes(rate)}/s" if rate else ""
            log_print(
                f"{desc}: {pct}% "
                f"({_format_bytes(self.n)}/{_format_bytes(self.total)}){rate_str}"
            )
        elif self.n > 0:
            log_print(f"{desc}: {_format_bytes(self.n)} 已下载")

        self._last_log_time = now


def log_byte_copy(
    *,
    src,
    dst,
    label: str,
    log_interval: float = 15.0,
    chunk_size: int = 64 * 1024 * 1024,
) -> int:
    """复制文件并周期性打印进度，返回写入字节数。"""
    total = src.stat().st_size
    written = 0
    last_log = time.time()
    log_print(f"{label}: 开始 ({_format_bytes(total)})")

    with open(src, "rb") as in_f, open(dst, "ab") as out_f:
        while chunk := in_f.read(chunk_size):
            out_f.write(chunk)
            written += len(chunk)
            now = time.time()
            if now - last_log >= log_interval or written >= total:
                pct = 100 * written / total
                log_print(f"{label}: {pct:.1f}% ({_format_bytes(written)}/{_format_bytes(total)})")
                last_log = now

    log_print(f"{label}: 完成 ({_format_bytes(written)})")
    return written
