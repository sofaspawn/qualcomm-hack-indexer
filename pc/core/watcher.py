"""File system watcher for the Lore indexing pipeline.

Monitors directories for file changes and triggers indexing operations
through the ``IndexManager``. Uses watchdog with per-file debouncing to
handle rapid filesystem events gracefully.

Usage:
    from pc.core.watcher import FileWatcher, watch

    # Object-oriented usage
    watcher = FileWatcher(index_manager, Path("./documents"))
    watcher.start()
    # ... later ...
    watcher.stop()

    # Blocking convenience function
    watch(Path("./documents"), index_manager)  # Ctrl-C to stop
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Timer

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from pc.config import DEBOUNCE_SECONDS, SUPPORTED_EXTENSIONS
from pc.core.index_manager import IndexManager

logger = logging.getLogger(__name__)

# Patterns to ignore.
_TEMP_SUFFIXES: set[str] = {".tmp", ".swp", ".swo", ".bak", ".crdownload", ".part"}
_TEMP_PREFIXES: tuple[str, ...] = ("~", ".~")


class _DebouncedHandler(FileSystemEventHandler):
    """Filesystem event handler with per-file debouncing.

    Buffers rapid events for the same file and only triggers indexing
    after a quiet period (``debounce_seconds``). Ignores hidden files,
    temp files, and unsupported extensions.

    Args:
        index_manager: The ``IndexManager`` to delegate indexing to.
        debounce_seconds: Quiet period before triggering.
        supported_extensions: Set of extensions to process.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        supported_extensions: set[str] = SUPPORTED_EXTENSIONS,
    ) -> None:
        super().__init__()
        self._index_manager = index_manager
        self._debounce_seconds = debounce_seconds
        self._supported_extensions = supported_extensions
        self._timers: dict[str, Timer] = {}

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: The filesystem event.
        """
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: The filesystem event.
        """
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events.

        Args:
            event: The filesystem event.
        """
        if not event.is_directory:
            self._handle_delete(event.src_path)

    def _should_ignore(self, file_path: str) -> bool:
        """Check if a file should be ignored.

        Args:
            file_path: Absolute path to the file.

        Returns:
            True if the file should be skipped.
        """
        path = Path(file_path)
        name = path.name

        # Hidden files.
        if name.startswith("."):
            return True

        # Temp file prefixes.
        if name.startswith(_TEMP_PREFIXES):
            return True

        # Temp file suffixes.
        if path.suffix.lower() in _TEMP_SUFFIXES:
            return True

        # Unsupported extensions.
        extension = path.suffix.lower().lstrip(".")
        if extension not in self._supported_extensions:
            return True

        return False

    def _handle_change(self, file_path: str) -> None:
        """Debounce a create/modify event and schedule indexing.

        Args:
            file_path: Absolute path to the changed file.
        """
        if self._should_ignore(file_path):
            return

        # Cancel any pending timer for this file.
        existing = self._timers.pop(file_path, None)
        if existing is not None:
            existing.cancel()

        # Schedule indexing after the debounce period.
        timer = Timer(
            self._debounce_seconds,
            self._index_with_retry,
            args=(file_path,),
        )
        timer.daemon = True
        self._timers[file_path] = timer
        timer.start()

        logger.debug("Debounce timer set for '%s' (%.1fs)", file_path, self._debounce_seconds)

    def _handle_delete(self, file_path: str) -> None:
        """Handle a file deletion event.

        Args:
            file_path: Absolute path to the deleted file.
        """
        if self._should_ignore(file_path):
            return

        # Cancel any pending indexing timer for this file.
        existing = self._timers.pop(file_path, None)
        if existing is not None:
            existing.cancel()

        logger.info("File deleted: '%s'", file_path)
        try:
            self._index_manager.delete_document(file_path)
        except Exception:
            logger.exception("Failed to delete document for '%s'", file_path)

    def _index_with_retry(self, file_path: str) -> None:
        """Index a file, waiting for writes to complete.

        Checks that the file exists and its size is stable before
        triggering indexing.

        Args:
            file_path: Absolute path to the file.
        """
        self._timers.pop(file_path, None)
        path = Path(file_path)

        if not path.exists():
            logger.warning("File no longer exists after debounce: '%s'", file_path)
            return

        # Wait for the file to finish writing (size stability check).
        if not _wait_for_stable_size(path):
            logger.warning("File size unstable, skipping: '%s'", file_path)
            return

        logger.info("Watcher triggering index for: '%s'", file_path)
        try:
            self._index_manager.reindex_file(file_path)
        except Exception:
            logger.exception("Watcher failed to index '%s'", file_path)

    def cancel_all(self) -> None:
        """Cancel all pending debounce timers."""
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()


class FileWatcher:
    """High-level file watcher with start/stop lifecycle.

    Wraps a watchdog ``Observer`` and a ``_DebouncedHandler`` to monitor
    a directory tree for supported document changes.

    Args:
        index_manager: The ``IndexManager`` to delegate indexing to.
        directory: Root directory to watch.
        recursive: Whether to watch subdirectories.
        debounce_seconds: Quiet period before triggering indexing.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        directory: str | Path,
        recursive: bool = True,
        debounce_seconds: float = DEBOUNCE_SECONDS,
    ) -> None:
        self._directory = Path(directory).resolve()
        self._recursive = recursive
        self._handler = _DebouncedHandler(
            index_manager=index_manager,
            debounce_seconds=debounce_seconds,
        )
        self._observer = Observer()

    def start(self) -> None:
        """Start watching the directory for file changes."""
        if not self._directory.is_dir():
            raise FileNotFoundError(
                f"Watch directory does not exist: '{self._directory}'"
            )

        self._observer.schedule(
            self._handler,
            str(self._directory),
            recursive=self._recursive,
        )
        self._observer.start()
        logger.info(
            "Watcher started: '%s' (recursive=%s)",
            self._directory,
            self._recursive,
        )

    def stop(self) -> None:
        """Stop watching and clean up resources."""
        logger.info("Watcher stopping: '%s'", self._directory)
        self._handler.cancel_all()
        self._observer.stop()
        self._observer.join()
        logger.info("Watcher stopped")


def watch(
    directory: str | Path,
    index_manager: IndexManager,
    recursive: bool = True,
    debounce_seconds: float = DEBOUNCE_SECONDS,
) -> None:
    """Convenience function to watch a directory (blocking).

    Runs until interrupted with Ctrl-C.

    Args:
        directory: Root directory to watch.
        index_manager: The ``IndexManager`` to delegate indexing to.
        recursive: Whether to watch subdirectories.
        debounce_seconds: Quiet period before triggering indexing.
    """
    watcher = FileWatcher(
        index_manager=index_manager,
        directory=directory,
        recursive=recursive,
        debounce_seconds=debounce_seconds,
    )
    watcher.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received — shutting down watcher")
    finally:
        watcher.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_stable_size(
    path: Path, checks: int = 3, interval: float = 0.5
) -> bool:
    """Wait until a file's size stops changing.

    Args:
        path: Path to the file.
        checks: Number of consecutive stable size readings required.
        interval: Seconds between size checks.

    Returns:
        True if the file size stabilized, False if the file disappeared
        or remained unstable.
    """
    previous_size: int | None = None
    stable_count = 0

    for _ in range(checks * 2):
        if not path.exists():
            return False

        try:
            current_size = path.stat().st_size
        except OSError:
            return False

        if current_size == previous_size:
            stable_count += 1
            if stable_count >= checks:
                return True
        else:
            stable_count = 0

        previous_size = current_size
        time.sleep(interval)

    return stable_count >= checks
