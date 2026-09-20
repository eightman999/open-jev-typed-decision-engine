"""One decide() interface, swappable backend.

  engine = DecisionEngine(backend="coreml", compute_units="cpu_and_ne")
  result = engine.decide(state, questions)

backend="torch" delegates to the reference PyTorch implementation in
05_serve.py unchanged - it stays the parity baseline. backend="coreml" runs
encoder+head on Core ML (CPU_AND_NE by default) and keeps the dynamic-schema
bookkeeping on CPU.
"""
from __future__ import annotations
import importlib.util


def _serve_module():
    """05_serve.py starts with a digit - import it by path like decide.py."""
    spec = importlib.util.spec_from_file_location("serve", "05_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _TorchEngine:
    """Thin wrapper over the reference implementation - do not reimplement it."""
    def __init__(self, ckpt, device):
        self._serve = _serve_module()
        self.info = self._serve.load(ckpt, device)

    def decide(self, state, questions, calibrated=True):
        return self._serve.decide(state, questions, calibrated)


class DecisionEngine:
    def __init__(self, backend="torch", ckpt="jevlite.pt", device=None,
                 compute_units="cpu_and_ne", pkg_dir="artifacts/coreml"):
        self.backend = backend
        if backend == "torch":
            self._impl = _TorchEngine(ckpt, device)
            self.dev = self._impl.info["dev"]
        elif backend == "coreml":
            from coreml_backend import CoreMLEngine
            self._impl = CoreMLEngine(pkg_dir, ckpt=ckpt,
                                      compute_units=compute_units)
            self.dev = f"coreml:{compute_units}"
        else:
            raise ValueError(f"unknown backend {backend!r} - "
                             "use 'torch' or 'coreml'")

    def decide(self, state, questions, calibrated=True):
        """-> {question: {"label", "confidence", "probabilities"}}"""
        return self._impl.decide(state, questions, calibrated)
