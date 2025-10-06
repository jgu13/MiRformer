# scripts/ckpt_utils.py
import os, re, io, json, time, torch, random, signal
import numpy as np

def is_rank0():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0 and int(os.environ.get("RANK", "0")) == 0

def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    with io.BytesIO() as buf:
        torch.save(obj, buf)
        data = buf.getvalue()
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def list_checkpoints(dirpath, pattern=r".*\.pth$"):
    if not os.path.isdir(dirpath): return []
    rgx = re.compile(pattern)
    return sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath) if rgx.match(f)])

def latest_checkpoint(dirpath):
    cks = list_checkpoints(dirpath)
    if not cks: return None
    # newest by mtime
    return max(cks, key=os.path.getmtime)

def save_training_state(
    model, optimizer, scheduler, epoch, best_metrics: dict,
    ckpt_path
):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_metrics": best_metrics,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, ckpt_path)
    print(f"[CKPT] saved: {ckpt_path}", flush=True)
    return ckpt_path

def load_training_state(ckpt_path, model, optimizer=None, scheduler=None, map_location=None):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return {"epoch": 0, "step": 0, "best_metrics": {}}
    ckpt = torch.load(ckpt_path, map_location=map_location or "cpu")
    # model first
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[CKPT] loaded with missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    # the rest if provided
    # if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    # if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    # RNG
    try:
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.random.set_rng_state(ckpt["rng"]["torch_cpu"])
        if torch.cuda.is_available() and ckpt["rng"]["torch_cuda_all"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda_all"])
    except Exception as e:
        print(f"[CKPT] RNG restore warning: {e}", flush=True)

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_metrics": ckpt.get("best_metrics", {}),
        "path": ckpt_path,
    }

class GracefulKiller:
    """Catch SIGTERM/SIGINT to save and exit cleanly before walltime."""
    def __init__(self, on_kill):
        self.on_kill = on_kill
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT,  self._handler)
        self.triggered = False
    def _handler(self, signum, frame):
        if not self.triggered:
            self.triggered = True
            print(f"[CKPT] Caught signal {signum}, saving and exiting...", flush=True)
            try: self.on_kill()
            finally: os._exit(0)
