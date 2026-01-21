"""
vLLM utilities for robust engine lifecycle management.

Ensures proper shutdown of vLLM EngineCore processes to prevent hangs
in multi-run evaluation loops.
"""

import atexit
import gc
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from typing import Optional, List, Any

import torch

# Set tokenizers parallelism BEFORE any imports that might trigger HF tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Track active vLLM engines for emergency cleanup
_ACTIVE_ENGINES: List[Any] = []
_CLEANUP_LOCK = threading.Lock()


def _get_engine_processes(llm) -> List[BaseProcess]:
    """
    Extract all EngineCore processes from a vLLM LLM instance.

    Args:
        llm: vLLM LLM instance

    Returns:
        List of subprocess objects
    """
    procs = []
    try:
        # Path: llm.llm_engine.engine_core.resources.engine_manager.processes
        engine = getattr(llm, 'llm_engine', None)
        if engine is None:
            return procs

        engine_core = getattr(engine, 'engine_core', None)
        if engine_core is None:
            return procs

        # For MPClient (sync)
        resources = getattr(engine_core, 'resources', None)
        if resources is not None:
            engine_manager = getattr(resources, 'engine_manager', None)
            if engine_manager is not None:
                procs.extend(getattr(engine_manager, 'processes', []))

        # Direct engine_manager on engine_core
        engine_manager = getattr(engine_core, 'engine_manager', None)
        if engine_manager is not None:
            procs.extend(getattr(engine_manager, 'processes', []))

    except Exception as e:
        print(f"[vLLM Cleanup] Warning: Could not extract processes: {e}")

    return procs


def _kill_process_tree(pid: int):
    """Kill a process and all its children."""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
    except ImportError:
        # Fallback without psutil
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _force_shutdown_processes(procs: List[BaseProcess], timeout: float = 5.0):
    """
    Force shutdown of processes with timeout.

    Args:
        procs: List of processes to terminate
        timeout: Seconds to wait before force kill
    """
    # First, try graceful terminate
    for proc in procs:
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass

    # Wait with timeout
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            try:
                proc.join(max(0.1, remaining))
            except Exception:
                pass

    # Force kill any remaining
    for proc in procs:
        if proc.is_alive():
            pid = proc.pid
            if pid is not None:
                print(f"[vLLM Cleanup] Force killing process {pid}")
                _kill_process_tree(pid)


def shutdown_vllm_engine(llm, verbose: bool = True) -> bool:
    """
    Properly shutdown a vLLM LLM engine and all its subprocesses.

    This function ensures complete cleanup of:
    - EngineCore subprocesses
    - ZMQ sockets
    - GPU memory
    - CUDA contexts

    Args:
        llm: vLLM LLM instance
        verbose: Print diagnostic messages

    Returns:
        True if shutdown was successful
    """
    if llm is None:
        return True

    pid = os.getpid()
    if verbose:
        print(f"[vLLM Cleanup] Shutting down engine from PID {pid}")

    success = True
    procs = []

    try:
        # Get processes before we start cleanup
        procs = _get_engine_processes(llm)
        if verbose and procs:
            print(f"[vLLM Cleanup] Found {len(procs)} EngineCore processes: {[p.pid for p in procs if p.pid]}")

        # Try to call shutdown via engine_core
        try:
            engine = getattr(llm, 'llm_engine', None)
            if engine is not None:
                engine_core = getattr(engine, 'engine_core', None)
                if engine_core is not None:
                    shutdown_fn = getattr(engine_core, 'shutdown', None)
                    if shutdown_fn is not None:
                        if verbose:
                            print(f"[vLLM Cleanup] Calling engine_core.shutdown()")
                        shutdown_fn()
        except Exception as e:
            if verbose:
                print(f"[vLLM Cleanup] engine_core.shutdown() failed: {e}")
            success = False

        # Delete the LLM object to trigger __del__ and finalizers
        try:
            del llm
        except Exception:
            pass

        # Force garbage collection to trigger finalizers
        gc.collect()
        gc.collect()

        # Give finalizers time to run
        time.sleep(0.5)

        # Check if processes are still alive and force kill
        still_alive = [p for p in procs if p.is_alive()]
        if still_alive:
            if verbose:
                print(f"[vLLM Cleanup] {len(still_alive)} processes still alive, force killing...")
            _force_shutdown_processes(still_alive, timeout=3.0)

        # Final check
        still_alive = [p for p in procs if p.is_alive()]
        if still_alive:
            if verbose:
                print(f"[vLLM Cleanup] WARNING: {len(still_alive)} processes still alive after force kill!")
            success = False

    except Exception as e:
        if verbose:
            print(f"[vLLM Cleanup] Error during shutdown: {e}")
            traceback.print_exc()
        success = False

    finally:
        # Always try to clean up GPU memory
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    if verbose:
        print(f"[vLLM Cleanup] Shutdown {'complete' if success else 'completed with warnings'}")

    return success


@contextmanager
def vllm_engine_context(llm_kwargs: dict, verbose: bool = True):
    """
    Context manager for vLLM LLM engine with guaranteed cleanup.

    Usage:
        with vllm_engine_context({"model": "meta-llama/Llama-3.1-8B", ...}) as llm:
            outputs = llm.generate(prompts, sampling_params)

    Args:
        llm_kwargs: Keyword arguments for vLLM LLM constructor
        verbose: Print diagnostic messages

    Yields:
        vLLM LLM instance
    """
    from vllm import LLM

    llm = None
    pid = os.getpid()

    if verbose:
        print(f"[vLLM] Creating engine from PID {pid}")
        print(f"[vLLM] Config: {llm_kwargs}")

    try:
        llm = LLM(**llm_kwargs)

        # Track for emergency cleanup
        with _CLEANUP_LOCK:
            _ACTIVE_ENGINES.append(llm)

        if verbose:
            procs = _get_engine_processes(llm)
            print(f"[vLLM] Engine created with {len(procs)} EngineCore processes")

        yield llm

    finally:
        # Remove from tracking
        with _CLEANUP_LOCK:
            if llm in _ACTIVE_ENGINES:
                _ACTIVE_ENGINES.remove(llm)

        # Shutdown
        shutdown_vllm_engine(llm, verbose=verbose)


def cleanup_all_vllm_engines(verbose: bool = True):
    """
    Emergency cleanup of all tracked vLLM engines.
    Call this at process exit or on error.
    """
    with _CLEANUP_LOCK:
        engines = list(_ACTIVE_ENGINES)
        _ACTIVE_ENGINES.clear()

    for llm in engines:
        try:
            shutdown_vllm_engine(llm, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[vLLM Cleanup] Error cleaning up engine: {e}")


def kill_vllm_processes_by_name(process_name: str = "EngineCore", verbose: bool = True):
    """
    Find and kill any orphaned vLLM EngineCore processes.

    Use this as a last resort cleanup.

    Args:
        process_name: Name pattern to match
        verbose: Print what's being killed
    """
    try:
        import psutil
        current_pid = os.getpid()

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Skip our own process
                if proc.pid == current_pid:
                    continue

                # Check if it's a vLLM EngineCore process
                name = proc.info.get('name', '')
                cmdline = ' '.join(proc.info.get('cmdline', []) or [])

                if process_name in name or 'vllm' in cmdline.lower():
                    if verbose:
                        print(f"[vLLM Cleanup] Killing orphaned process: PID={proc.pid} name={name}")
                    proc.kill()

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except ImportError:
        if verbose:
            print("[vLLM Cleanup] psutil not available, cannot scan for orphaned processes")


# Register cleanup on exit
atexit.register(cleanup_all_vllm_engines, verbose=False)


class WatchdogTimer:
    """
    Watchdog timer that dumps stack traces if evaluation exceeds timeout.
    """

    def __init__(self, timeout_seconds: float, callback=None):
        self.timeout = timeout_seconds
        self.callback = callback or self._default_callback
        self._timer = None

    def _default_callback(self):
        print("\n" + "=" * 60)
        print(f"WATCHDOG TIMEOUT ({self.timeout}s) - Dumping stack traces")
        print("=" * 60)

        # Print stack traces of all threads
        for thread_id, frame in sys._current_frames().items():
            print(f"\nThread {thread_id}:")
            traceback.print_stack(frame)

        # Print vLLM process info
        try:
            import psutil
            print("\n--- vLLM Processes ---")
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
                try:
                    cmdline = ' '.join(proc.info.get('cmdline', []) or [])
                    if 'vllm' in cmdline.lower() or 'EngineCore' in proc.info.get('name', ''):
                        mem = proc.info.get('memory_info')
                        mem_mb = mem.rss / 1024 / 1024 if mem else 0
                        print(f"  PID={proc.pid} name={proc.info.get('name')} mem={mem_mb:.0f}MB")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass

        print("=" * 60)

    def start(self):
        self._timer = threading.Timer(self.timeout, self.callback)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.cancel()
