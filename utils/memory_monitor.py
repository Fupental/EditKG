import time

import torch


def _fmt_gb(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f}GB"


def cuda_mem_stats(device=None):
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.cuda.current_device()
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        device = device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, str):
        device = torch.device(device)
        if device.type != "cuda":
            return None
        device = device.index if device.index is not None else torch.cuda.current_device()

    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
    }


def log_cuda_mem(tag, device=None, enabled=True, extra=None):
    if not enabled:
        return
    stats = cuda_mem_stats(device)
    if stats is None:
        print(f"[MEM] {tag} | cuda_unavailable")
        return

    msg = (
        f"[MEM] {tag} | "
        f"alloc={_fmt_gb(stats['allocated'])} "
        f"reserved={_fmt_gb(stats['reserved'])} "
        f"max_alloc={_fmt_gb(stats['max_allocated'])} "
        f"max_reserved={_fmt_gb(stats['max_reserved'])}"
    )
    if extra:
        msg += f" | {extra}"
    print(msg)


class MemTimer:
    def __init__(self, tag, device=None, enabled=True):
        self.tag = tag
        self.device = device
        self.enabled = enabled
        self.start = None

    def __enter__(self):
        self.start = time.time()
        log_cuda_mem(f"{self.tag}:start", self.device, self.enabled)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start if self.start is not None else 0.0
        log_cuda_mem(f"{self.tag}:end", self.device, self.enabled, extra=f"elapsed={elapsed:.2f}s")
