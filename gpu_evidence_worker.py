"""Long-running GPU evidence worker.

Reads JSON requests from stdin (one per line), collects GPU evidence,
writes JSON responses to stdout (one per line).

Protocol:
  Request:  {"nonce": "<hex>", "no_gpu_mode": false}
  Response: {"ok": true, "evidence": [...]}
  Error:    {"ok": false, "error": "message"}

Keeps the Python interpreter, verifier module, and NVML driver initialized
across requests, avoiding ~0.5-2s startup overhead per call.
"""

import json
import sys
import traceback

# Import verifier once at startup — this is the expensive part
# (loads shared libraries, may trigger nvmlInit on import).
try:
    from verifier import cc_admin
    IMPORT_OK = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_OK = False
    IMPORT_ERROR = str(e)


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
    # Signal readiness to the Rust parent.
    ready_msg = {"ready": True, "import_ok": IMPORT_OK}
    if not IMPORT_OK:
        ready_msg["import_error"] = IMPORT_ERROR
    sys.stdout.write(json.dumps(ready_msg) + "\n")
    sys.stdout.flush()

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

        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
