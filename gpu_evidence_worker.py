"""Long-running GPU evidence worker.

Reads JSON requests from stdin (one per line), collects GPU evidence,
writes JSON responses to stdout (one per line).

Protocol:
  Request:  {"nonce": "<hex>", "no_gpu_mode": false}
  Response: {"ok": true, "evidence": [...]}
  Error:    {"ok": false, "error": "message"}

Keeps the Python interpreter, verifier module, and NVML driver initialized
across requests, avoiding ~0.5-2s startup overhead per call.

IMPORTANT: The NVIDIA verifier library (cc_admin) prints info messages
directly to stdout (e.g. "Number of GPUs available : 8"). We redirect
stdout to /dev/null during evidence collection and use a saved reference
to the real stdout for our JSON protocol.
"""

import io
import json
import os
import sys
import traceback

# Save the real stdout fd before anything can pollute it.
# We dup the fd so even if sys.stdout is replaced, we can still write.
_real_stdout_fd = os.dup(sys.stdout.fileno())
_real_stdout = os.fdopen(_real_stdout_fd, "w", buffering=1)  # line-buffered

# Redirect sys.stdout to stderr so any library prints go to stderr
# (which the Rust parent reads separately / ignores).
sys.stdout = sys.stderr

# Import verifier once at startup — this is the expensive part
# (loads shared libraries, may trigger nvmlInit on import).
try:
    from verifier import cc_admin
    IMPORT_OK = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_OK = False
    IMPORT_ERROR = str(e)


def _write_response(obj):
    """Write a JSON response to the real stdout (not the redirected one)."""
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def collect(nonce_hex: str, no_gpu_mode: bool):
    """Collect GPU evidence for the given nonce."""
    if not IMPORT_OK:
        return {"ok": False, "error": f"verifier import failed: {IMPORT_ERROR}"}
    try:
        if no_gpu_mode:
            evidence = cc_admin.collect_gpu_evidence_remote(
                nonce_hex, no_gpu_mode=True
            )
        else:
            evidence = cc_admin.collect_gpu_evidence_remote(
                nonce_hex, ppcie_mode=False
            )
        return {"ok": True, "evidence": evidence}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main():
    # Signal readiness to the Rust parent on the real stdout.
    ready_msg = {"ready": True, "import_ok": IMPORT_OK}
    if not IMPORT_OK:
        ready_msg["import_error"] = IMPORT_ERROR
    _write_response(ready_msg)

    # Read requests from stdin (which is NOT redirected).
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            nonce_hex = request["nonce"]
            no_gpu_mode = request.get("no_gpu_mode", False)
            result = collect(nonce_hex, no_gpu_mode)
        except json.JSONDecodeError as e:
            result = {"ok": False, "error": f"invalid JSON: {e}"}
        except KeyError as e:
            result = {"ok": False, "error": f"missing field: {e}"}
        except Exception as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}

        _write_response(result)


if __name__ == "__main__":
    main()
