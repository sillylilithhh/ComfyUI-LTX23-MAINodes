"""LTX-2.3 adaptation of the three MAINodes motion nodes.

The original implementation targets MiniMax/H3's 17k+5 temporal grid and
24-channel latent layout.  LTX-2.3 instead uses a 128-channel video latent
with temporal VAE compression of 8x, so legal video lengths are 8k+1.

Only the three nodes requested for LTX-2.3 are included here:
    LTX23JerkOracle
    LTX23TimeSmear
    LTX23ExactRecover

The algorithm itself is intentionally kept close to the original MAINodes
implementation: jerk/trajectory profile -> quantile threshold -> ramped hold
map -> integer frame smear -> exact first-frame recovery.
"""

import json
import math

import numpy as np
import torch

LTX23_LATENT_CHANNELS = 128
LTX23_TEMPORAL_SCALE = 8
LTX23_MIN_LEGAL_FRAMES = 9

# Cost model is retained from the original nodes because it is only a report
# and does not affect the actual node outputs.
COST_EXP = 1.7
OVERHEAD_S = 6.7

PROFILE_MODES = {
    "value |d3| (default)": ("value", 3),
    "value |d1| (energy baseline)": ("value", 1),
    "trajectory centroid |d3|": ("traj", 3),
}


def _video_component(samples):
    """Return a plain LTX video latent tensor shaped (B, 128, T, H, W).

    LTX-2.3 video latents are standard ComfyUI LATENT dictionaries with
    128 channels and 8x temporal compression.  We accept a nested tensor if
    one is supplied, matching the original node's tolerance for nested data.
    """
    if not isinstance(samples, dict) or "samples" not in samples:
        raise ValueError("Expected a LATENT containing a 'samples' tensor")
    z = samples["samples"]
    if hasattr(z, "is_nested") and z.is_nested:
        z = z.tensors[0]
    if not torch.is_tensor(z) or z.ndim != 5:
        raise ValueError(
            f"Expected a 5D LTX video latent [B, C, T, H, W], got {type(z).__name__} "
            f"with shape {getattr(z, 'shape', None)}"
        )
    if z.shape[1] != LTX23_LATENT_CHANNELS:
        raise ValueError(
            f"This node is for LTX-2.3 video latents: expected 128 channels, "
            f"got {z.shape[1]}."
        )
    return z


def _value_profile(v, order):
    """Per-token |Δ^order| of latent values, averaged over C/H/W."""
    j = np.abs(np.diff(v, n=order, axis=2)).mean(axis=(0, 1, 3, 4))
    lead = order // 2
    return np.pad(j, (lead, v.shape[2] - len(j) - lead), mode="edge")


def _trajectory_profile(v, order=3):
    """Per-token derivative magnitude of the latent energy centroid path."""
    e = np.abs(v).mean(axis=(0, 1))  # (T, H, W)
    e = e - e.min(axis=(1, 2), keepdims=True)
    tot = e.sum(axis=(1, 2)) + 1e-8
    ys = np.arange(e.shape[1], dtype=np.float64)[None, :, None]
    xs = np.arange(e.shape[2], dtype=np.float64)[None, None, :]
    cy = (e * ys).sum(axis=(1, 2)) / tot
    cx = (e * xs).sum(axis=(1, 2)) / tot
    path = np.stack([cy, cx], axis=1)
    j = np.linalg.norm(np.diff(path, n=order, axis=0), axis=1)
    lead = order // 2
    return np.pad(j, (lead, path.shape[0] - len(j) - lead), mode="edge")


def _phase_norm(prof):
    # Kept verbatim in spirit from the original implementation.  LTX has a
    # uniform temporal latent clock, so this normalization is no longer
    # compensating for H3's (1,4,4,4,4) temporal phase structure.  It is
    # therefore optional here and disabled by default.
    return prof


def _jerk_profile(z, mode="value |d3| (default)", phase_norm=False):
    v = z.detach().float().cpu().numpy()
    kind, order = PROFILE_MODES.get(mode, ("value", 3))
    prof = _value_profile(v, order) if kind == "value" else _trajectory_profile(v, order)
    return _phase_norm(prof) if phase_norm else prof


def _legal_ceil(n):
    """Ceil a pixel-frame count to LTX's legal 8k+1 temporal grid."""
    n = max(1, int(n))
    if n <= 1:
        return 1
    if n <= LTX23_MIN_LEGAL_FRAMES:
        return LTX23_MIN_LEGAL_FRAMES
    return 1 + 8 * math.ceil((n - 1) / 8)


def _grid_token_count(frames):
    """Number of LTX video latent time positions for a legal frame count."""
    frames = _legal_ceil(frames)
    return ((frames - 1) // 8) + 1


def _frame_token(f, t_lat):
    """Map a pixel frame to its LTX latent temporal cell.

    LTX has one anchor frame followed by 8-frame intervals:
      token 0 -> frame 0
      token 1 -> frames 1..8
      token 2 -> frames 9..16
      ...
    """
    if t_lat <= 1:
        return 0
    return min((int(f) + 7) // 8, t_lat - 1)


def _latent_holds_to_frame_holds(latent_holds, n_frames):
    """Expand LTX latent-time hold values onto decoded IMAGE frames.

    LTX's temporal VAE clock is 1 + 8k: latent token 0 owns frame 0, and
    each following latent token owns the next 8 decoded frames.  The oracle
    naturally works on latent positions; Time Smear works on actual IMAGE
    frames.  This bridges the two clocks and crops the final block to the
    exact image-batch length when a downstream node is using a shorter clip.
    """
    latent_holds = [int(h) for h in latent_holds]
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    if not latent_holds or min(latent_holds) < 1:
        raise ValueError("latent hold counts must be positive")

    out = []
    for token, hold in enumerate(latent_holds):
        block_len = 1 if token == 0 else 8
        remaining = n_frames - len(out)
        if remaining <= 0:
            break
        out.extend([hold] * min(block_len, remaining))

    if len(out) < n_frames:
        # A short map can occur when a workflow crops after the oracle.
        # Continue the final known latent rate rather than failing.
        out.extend([latent_holds[-1]] * (n_frames - len(out)))

    return out


def _profile_to_plan(prof, length, q, d_max, ramp, bridge):
    prof = np.asarray(prof, dtype=np.float64)
    t_lat = len(prof)
    if t_lat == 0:
        raise ValueError("The LTX latent has no temporal positions")

    thr = np.quantile(prof, q)
    tok_d = np.where(prof >= thr, d_max, 1).astype(int)

    if bridge:
        hot = np.where(tok_d == d_max)[0]
        for a, b in zip(hot[:-1], hot[1:]):
            if 1 < b - a <= bridge:
                tok_d[a:b + 1] = d_max

    if ramp:
        for _ in range(d_max - 1):
            left = np.concatenate([[1], tok_d[:-1]])
            right = np.concatenate([tok_d[1:], [1]])
            tok_d = np.maximum(tok_d, np.maximum(left, right) - 1)

    holds = [int(tok_d[_frame_token(f, t_lat)]) for f in range(length)]

    # Segment output stays useful for inspection.  Segment units are LTX
    # latent-time positions rather than H3 positions.
    segs = []
    t0 = 0
    for t in range(1, t_lat + 1):
        if t == t_lat or tok_d[t] != tok_d[t0]:
            if tok_d[t0] > 1:
                segs.append(f"{t0}:{t}:{int(tok_d[t0])}")
            t0 = t

    hot = np.where(tok_d > 1)[0]
    if len(hot):
        w0 = _frame_token(8 * int(hot.min()), t_lat) * 8
        end_token = min(int(hot.max()) + 1, t_lat - 1)
        w1 = min(length, 8 * end_token + 1)
        wlen = max(0, w1 - w0)
    else:
        w0, wlen = 0, 0

    return holds, ",".join(segs), w0, wlen, tok_d


def _cost_report(world_len, dilated, fps=24, s_per_step=0.0, est_steps=18,
                 overhead_s=OVERHEAD_S, tail=""):
    world_len = max(1, int(world_len))
    dilated = max(world_len, int(dilated))
    fps = max(1, int(fps))
    t_world = _grid_token_count(world_len)
    t_dil = _grid_token_count(dilated)
    time_x = (t_dil / t_world) ** COST_EXP
    report = (
        f"{world_len}f ({world_len / fps:.1f}s) -> {dilated}f "
        f"({dilated / fps:.1f}s) effective regen, "
        f"{dilated / world_len:.2f}x frames / {time_x:.1f}x time per "
        f"step; LTX latent frames {t_world} -> {t_dil}"
    )
    if tail:
        report += f"; {tail}"
    if s_per_step > 0:
        secs = time_x * (overhead_s + s_per_step * max(1, int(est_steps)))
        report += (
            f"; roughly {secs / 60:.0f} min at {s_per_step:g} s/step x "
            f"{int(est_steps)} steps (+{overhead_s:g}s encode/decode, "
            f"all x{time_x:.1f})"
        )
    return report


def _cost_widgets(with_fps=False):
    w = {}
    if with_fps:
        w["fps"] = ("INT", {
            "default": 24,
            "min": 1,
            "max": 120,
            "tooltip": "Only used to phrase the report in seconds.",
        })
    w["s_per_step"] = ("FLOAT", {
        "default": 0.0,
        "min": 0.0,
        "max": 3600.0,
        "step": 0.1,
        "tooltip": "Optional baseline sampler seconds/step used only for the estimate.",
    })
    w["est_steps"] = ("INT", {
        "default": 18,
        "min": 1,
        "max": 200,
        "tooltip": "Optional number of sampler steps used only for the estimate.",
    })
    w["overhead_s"] = ("FLOAT", {
        "default": OVERHEAD_S,
        "min": 0.0,
        "max": 600.0,
        "step": 0.1,
        "tooltip": "Optional encode/decode overhead used only for the estimate.",
    })
    return w


def expand_hold_map_to_end(holds):
    """Same end-jump fix as the original, adapted to the LTX 8-frame clock."""
    holds = [int(h) for h in holds]
    if not holds or min(holds) < 1:
        raise ValueError("hold counts must be positive")

    # For LTX, a tail up to one temporal block (8 frames) is treated as the
    # possible end jump. Longer tails are left as intentional quiet/rest.
    max_end_tail = LTX23_TEMPORAL_SCALE
    n = len(holds)
    tail = 0
    while tail < n and holds[n - 1 - tail] == 1:
        tail += 1
    if tail == 0 or tail == n or tail > max_end_tail:
        return holds, None

    start = n - tail - 1
    rate = holds[start]
    while start > 0 and holds[start - 1] == rate:
        start -= 1

    if rate <= 1:
        return holds, None

    out = holds[:]
    for i in range(start, n):
        if out[i] == 1:
            out[i] = rate

    target = _legal_ceil(sum(out))
    deficit = target - sum(out)
    if deficit > 0:
        # Add the legal-grid padding inside the lifted end region.
        i = n - 1
        while deficit > 0:
            out[i] += 1
            deficit -= 1
            i -= 1
            if i < start:
                i = n - 1

    note = (
        f"LTX2.3 end-tail expansion: lifted a {tail}-frame rate-1 tail "
        f"to x{rate} and snapped the result to the 8k+1 grid"
    )
    return out, note


class LTX23JerkOracle:
    DESCRIPTION = (
        "LTX-2.3 adaptation of H3 Jerk Oracle. Reads the LTX video latent "
        "(128 channels, 8x temporal VAE compression) and emits a source-frame "
        "hold map by projecting the latent motion profile back onto LTX's "
        "8k+1 video-frame clock. The actual profile and hold compiler remain "
        "the original algorithm."
    )

    PRESETS = {
        "balanced (default)": {"q": 0.75, "d_max": 4, "ramp": True},
        "max quality (wide plateau)": {"q": 0.70, "d_max": 4, "ramp": True},
        "economy (tight spans)": {"q": 0.85, "d_max": 3, "ramp": True},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "length": ("INT", {
                    "default": 97,
                    "min": 1,
                    "max": 3600,
                    "step": 8,
                    "tooltip": "Pixel-frame count in the source video; LTX legal lengths are 8k+1.",
                }),
                "q": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.5,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "Jerk quantile that counts as hot.",
                }),
                "d_max": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "tooltip": "Peak hold count on the hottest LTX latent positions.",
                }),
                "ramp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "C1 ramp shoulders instead of hard steps.",
                }),
            },
            "optional": {
                "preset": (["custom"] + list(cls.PRESETS), {
                    "default": "balanced (default)",
                    "tooltip": "Any preset overrides q, d_max, and ramp.",
                }),
                "bridge": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Fill short valleys between hot latent positions.",
                }),
                "profile_mode": (list(PROFILE_MODES), {
                    "default": "value |d3| (default)",
                }),
                "abstain_below": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Absolute peak-to-mean contrast gate; 0 disables it.",
                }),
                **_cost_widgets(with_fps=True),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "segments", "window_start", "window_len", "profile", "report")
    FUNCTION = "read"
    CATEGORY = "latent/ltx23/motion"

    def read(self, samples, length, q, d_max, ramp, preset="custom", bridge=8,
             profile_mode="value |d3| (default)", abstain_below=0.0,
             fps=24, s_per_step=0.0, est_steps=18, overhead_s=OVERHEAD_S):
        if preset in self.PRESETS:
            p = self.PRESETS[preset]
            q, d_max, ramp = p["q"], p["d_max"], p["ramp"]

        z = _video_component(samples)
        t_lat = z.shape[2]
        expected_min = _grid_token_count(length)
        if t_lat != expected_min:
            # The oracle can still run with the supplied latent, but the user
            # is very likely wiring the wrong frame count. Fail early with a
            # useful LTX-specific message rather than silently misaligning the
            # generated hold map.
            raise ValueError(
                f"LTX23JerkOracle length={length} expects {expected_min} temporal "
                f"latent positions, but the connected latent has {t_lat}. "
                f"Use the source video's actual frame count (legal LTX lengths are 8k+1)."
            )

        length = int(length)
        prof = _jerk_profile(z, profile_mode, phase_norm=False)
        contrast = float(prof.max() / max(prof.mean(), 1e-8))

        if abstain_below > 0.0 and contrast < abstain_below:
            flat = json.dumps({"holds": [1] * length, "world_len": int(length), "units": "frames"})
            return (
                flat,
                "",
                0,
                length,
                " ".join(f"{v:.2f}" for v in prof),
                _cost_report(
                    length,
                    _legal_ceil(length),
                    fps,
                    s_per_step,
                    est_steps,
                    overhead_s,
                    tail=f"abstained, profile contrast {contrast:.2f} < {abstain_below:g}",
                ),
            )

        holds, segs, w0, wlen, tok_d = _profile_to_plan(
            prof, length, q, d_max, ramp, bridge
        )
        # Project the latent-time profile back onto source/video frames.
        # This keeps the same contract as the original H3 oracle: one hold
        # value per source frame, which Time Smear can apply directly.
        hold_map = json.dumps({"holds": holds, "world_len": int(length), "units": "frames"})
        profile = " ".join(f"{v:.2f}" for v in prof)
        n_held = sum(1 for h in holds if h > 1)
        report = _cost_report(
            length,
            _legal_ceil(sum(holds)),
            fps,
            s_per_step,
            est_steps,
            overhead_s,
            tail=f"{n_held} of {length} frames held, peak x{int(tok_d.max())}",
        )
        return (hold_map, segs, int(w0), int(wlen), profile, report)


class LTX23TimeSmear:
    DESCRIPTION = (
        "LTX-2.3 adaptation of H3 Time Smear. Holds input IMAGE frames on an "
        "integer retime map and snaps the resulting sequence to LTX's legal "
        "8k+1 frame grid. Wire LTX23JerkOracle.hold_map for adaptive mode. "
        "Shorter/longer frame maps are automatically aligned to the incoming "
        "IMAGE batch; uncovered frames remain at native rate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dilation": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "tooltip": "Uniform hold count; ignored when hold_map is connected.",
                }),
            },
            "optional": {
                "hold_map": ("STRING", {
                    "default": "",
                    "tooltip": "hold_map from LTX23JerkOracle.",
                }),
                "expand_to_end": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Lift a short rate-1 tail behind an expansion span to reduce the end jump.",
                }),
                **_cost_widgets(with_fps=True),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "hold_map_used", "length", "report")
    FUNCTION = "smear"
    CATEGORY = "image/ltx23/motion"

    def smear(self, images, dilation, hold_map="", expand_to_end=True, fps=24,
              s_per_step=0.0, est_steps=18, overhead_s=OVERHEAD_S):
        images = images.detach().cpu()
        n = images.shape[0]
        if n < 1:
            raise ValueError("LTX23TimeSmear requires at least one image frame")

        map_units = "frames"
        alignment_note = None
        if hold_map.strip():
            data = json.loads(hold_map)
            holds = [int(h) for h in data["holds"]]
            map_units = data.get("units", "frames")

            if map_units in ("ltx_latent", "latent", "tokens"):
                # Explicit latent-time maps are supported for compatibility
                # with the early LTX23 build. Expand them to IMAGE frames.
                holds = _latent_holds_to_frame_holds(holds, n)
                alignment_note = "expanded explicit LTX latent hold map to image frames"
            elif len(holds) != n:
                # A hold map is fundamentally a source-frame map. If a
                # workflow crops/extends the IMAGE batch after the oracle,
                # keep the authored portion and leave newly uncovered frames
                # at native rate (hold=1) instead of crashing. This preserves
                # exact recovery for the map emitted by this smear node.
                original_len = len(holds)
                if original_len < n:
                    holds = holds + [1] * (n - original_len)
                    alignment_note = (
                        f"hold map covered {original_len} frames; appended "
                        f"{n - original_len} native-rate frames to match image batch"
                    )
                else:
                    holds = holds[:n]
                    alignment_note = (
                        f"hold map covered {original_len} frames; cropped it to "
                        f"the {n}-frame image batch"
                    )
        else:
            holds = [int(dilation)] * n

        if len(holds) != n:
            raise ValueError(f"hold map covers {len(holds)} frames after alignment, image batch has {n}")
        if any(int(h) < 1 for h in holds):
            raise ValueError("hold map entries must be positive integers")

        note = None
        if expand_to_end:
            holds, note = expand_hold_map_to_end(holds)

        target = _legal_ceil(sum(holds))
        n_held = sum(1 for h in holds if h > 1)
        holds = list(holds)
        holds[-1] += target - sum(holds)

        idx = torch.tensor(
            [i for i, h in enumerate(holds) for _ in range(int(h))],
            dtype=torch.long,
        )
        used = json.dumps({"holds": holds, "world_len": n, "units": "frames"})
        mode = (
            f"uniform x{dilation}"
            if not hold_map.strip()
            else (
                f"adaptive, {n_held} of {n} frames held"
                + (" (expanded from LTX latent map)" if map_units == "ltx_latent" else "")
            )
        )
        report = _cost_report(
            n, target, fps, s_per_step, est_steps, overhead_s, tail=mode
        )
        if note:
            report += "\n" + note
        if alignment_note:
            report += "\n" + alignment_note

        return (images[idx], used, int(target), report)


class LTX23ExactRecover:
    DESCRIPTION = (
        "LTX-2.3 adaptation of H3 Exact Recover. Keeps the first generated "
        "frame from every integer hold group, exactly inverting LTX23TimeSmear "
        "without interpolation or resampling."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "hold_map": ("STRING", {
                    "default": "",
                    "tooltip": "hold_map_used from the same LTX23TimeSmear node.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "recover"
    CATEGORY = "image/ltx23/motion"

    def recover(self, images, hold_map):
        if not hold_map.strip():
            raise ValueError("LTX23ExactRecover requires hold_map_used from LTX23TimeSmear")
        data = json.loads(hold_map)
        holds = [int(h) for h in data["holds"]]
        starts = []
        cur = 0
        for h in holds:
            if h < 1:
                raise ValueError("hold map entries must be positive integers")
            starts.append(cur)
            cur += h
        if cur != images.shape[0]:
            raise ValueError(
                f"hold map expands to {cur} image frames, but received {images.shape[0]}"
            )
        return (images[torch.tensor(starts, dtype=torch.long)].cpu(),)


NODE_CLASS_MAPPINGS = {
    "LTX23JerkOracle": LTX23JerkOracle,
    "LTX23TimeSmear": LTX23TimeSmear,
    "LTX23ExactRecover": LTX23ExactRecover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23JerkOracle": "LTX 2.3 Jerk Oracle",
    "LTX23TimeSmear": "LTX 2.3 Time Smear",
    "LTX23ExactRecover": "LTX 2.3 Exact Recover",
}
