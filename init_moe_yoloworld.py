"""Initialize a MoE-YOLOWorld (Conv-expert variant) from the official
``yolov8x-worldv2.pt`` checkpoint.

Mapping rules (paper §3.7.1, eq. (3.29)):
    * The 1x1 entry/exit convs of every C2f (``cv1``, ``cv2``)             -> copied
    * The first 3x3 conv of every Bottleneck   (``m.j.cv1``)               -> copied to ``bottlenecks.j.cv1``
    * The second 3x3 conv of every Bottleneck  (``m.j.cv2``)               -> copied to ``bottlenecks.j.shared_expert.conv``
                                                                              and to each ``bottlenecks.j.routing_experts.{0..Nr-1}.conv``
                                                                              with an additive Gaussian perturbation N(0, sigma^2 I)
    * Cross-modal gate / loss-free bias                                    -> kept random (paper §3.7.1)
    * Everything else (Conv stems, SPPF, head C2fAttn, WorldDetect, etc.)  -> shape-matched copy

The script is ``x``-scale by default (matches ``yolov8x-worldv2.pt``).
"""

import argparse
import os
import re
import sys
from copy import deepcopy

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ultralytics import YOLOWorld_MoE
from ultralytics.utils import LOGGER


# Backbone layer indices that contain the standard C2f in the original
# ``yolov8-worldv2.yaml`` (x scale = same indices).
BACKBONE_C2F_INDICES = (2, 4, 6, 8)


def _flatten_state(obj):
    """Return a state_dict whatever the on-disk format is."""
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise TypeError(f"Unsupported checkpoint object: {type(obj)}")


def load_pretrained_state(pt_path: str):
    """Load the official YOLO-Worldv2 checkpoint and return its state_dict."""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"] is not None:
        src = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt and ckpt["model"] is not None:
        src = ckpt["model"]
    else:
        src = ckpt
    return _flatten_state(src)


def transfer_weights(
    src_sd: dict,
    dst_model: torch.nn.Module,
    perturb_sigma: float = 0.01,
    seed: int = 0,
) -> dict:
    """Build a state_dict for ``dst_model`` from the original YOLO-World ``src_sd``.

    Returns
    -------
    stats : dict
        Counters describing what was copied / perturbed / kept random.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    dst_sd = dst_model.state_dict()
    new_sd = {k: v.clone() for k, v in dst_sd.items()}  # start from current init

    stats = dict(copied=0, perturbed=0, mismatched=0, kept_random=0)

    # ---- 1. Direct copy for every (key, shape) that matches ----------------
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            new_sd[k] = v.clone()
            stats["copied"] += 1

    # ---- 2. Per-Bottleneck remapping for the MoE C2f layers ---------------
    for li in BACKBONE_C2F_INDICES:
        # Find every Bottleneck index j present in src for this stage
        pat = re.compile(rf"^model\.{li}\.m\.(\d+)\.cv2\.conv\.weight$")
        bn_ids = sorted(int(m.group(1)) for m in (pat.match(k) for k in src_sd) if m)
        if not bn_ids:
            LOGGER.warning(f"[init-moe] no Bottleneck found at backbone[{li}] in pretrained ckpt; "
                           f"this stage will keep random init.")
            continue

        for j in bn_ids:
            # ---- 2a. Bottleneck.cv1 -> MoEBottleneckFinalConv.cv1 ----------
            for sub in ("conv.weight", "bn.weight", "bn.bias",
                        "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"):
                src_k = f"model.{li}.m.{j}.cv1.{sub}"
                dst_k = f"model.{li}.bottlenecks.{j}.cv1.{sub}"
                if src_k in src_sd and dst_k in dst_sd and src_sd[src_k].shape == dst_sd[dst_k].shape:
                    new_sd[dst_k] = src_sd[src_k].clone()
                    stats["copied"] += 1

            # ---- 2b. Bottleneck.cv2 -> shared_expert.conv (direct copy) ---
            for sub in ("conv.weight", "bn.weight", "bn.bias",
                        "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"):
                src_k = f"model.{li}.m.{j}.cv2.{sub}"
                dst_shared = f"model.{li}.bottlenecks.{j}.shared_expert.conv.{sub}"
                if src_k in src_sd and dst_shared in dst_sd and src_sd[src_k].shape == dst_sd[dst_shared].shape:
                    new_sd[dst_shared] = src_sd[src_k].clone()
                    stats["copied"] += 1

            # ---- 2c. Bottleneck.cv2 -> routing_experts.{i}.conv (copy + N(0, sigma^2)) ----
            # Discover Nr from destination model
            re_pat = re.compile(rf"^model\.{li}\.bottlenecks\.{j}\.routing_experts\.(\d+)\.conv\.conv\.weight$")
            nr = 0
            for k in dst_sd:
                m = re_pat.match(k)
                if m:
                    nr = max(nr, int(m.group(1)) + 1)

            for i in range(nr):
                base_src = f"model.{li}.m.{j}.cv2"
                base_dst = f"model.{li}.bottlenecks.{j}.routing_experts.{i}.conv"
                # Conv weight: copy + Gaussian perturbation
                src_w = src_sd.get(f"{base_src}.conv.weight")
                dst_w_key = f"{base_dst}.conv.weight"
                if src_w is not None and dst_w_key in dst_sd and src_w.shape == dst_sd[dst_w_key].shape:
                    noise = torch.empty_like(src_w).normal_(mean=0.0, std=perturb_sigma, generator=g)
                    new_sd[dst_w_key] = (src_w + noise).clone()
                    stats["perturbed"] += 1

                # BN params (gamma/beta and running stats): copy as-is so the
                # routing experts start from the same statistics as the shared
                # one and diverge through training.  This matches the paper's
                # "near-copy" intent for standard-conv experts (§3.7.1).
                for sub in ("bn.weight", "bn.bias",
                            "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"):
                    src_k = f"{base_src}.{sub}"
                    dst_k = f"{base_dst}.{sub}"
                    if src_k in src_sd and dst_k in dst_sd and src_sd[src_k].shape == dst_sd[dst_k].shape:
                        new_sd[dst_k] = src_sd[src_k].clone()
                        stats["copied"] += 1

    # ---- 3. Bookkeeping: what's still untouched (random init) -------------
    untouched = []
    for k in dst_sd:
        if torch.equal(new_sd[k], dst_sd[k]):
            untouched.append(k)
    stats["kept_random"] = len(untouched)

    # Sanity: shape mismatches that we could not handle
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape != v.shape:
            stats["mismatched"] += 1

    return new_sd, stats, untouched


def main():
    parser = argparse.ArgumentParser(description="Initialize MoE-YOLOWorld (Conv-expert) from yolov8x-worldv2.pt")
    parser.add_argument(
        "--src",
        default=os.path.join(ROOT, "weights/yolov8x-worldv2.pt"),
        help="Source pretrained YOLO-Worldv2 checkpoint",
    )
    parser.add_argument(
        "--cfg",
        default=os.path.join(ROOT, "yolov8x-worldv2-moe-final-conv.yaml"),
        help="YAML defining the MoE-YOLOWorld (Conv-expert) architecture",
    )
    parser.add_argument(
        "--dst",
        default=os.path.join(ROOT, "weights/MoE_final_conv_init_x.pt"),
        help="Output path for the initialized checkpoint",
    )
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Std of the Gaussian perturbation added to each routing expert (paper eq. 3.29)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every key that stays at random init")
    args = parser.parse_args()

    LOGGER.info(f"[init-moe-x] building target model from {args.cfg} ...")
    moe = YOLOWorld_MoE(args.cfg)

    LOGGER.info(f"[init-moe-x] loading source checkpoint {args.src} ...")
    src_sd = load_pretrained_state(args.src)

    LOGGER.info(f"[init-moe-x] transferring weights (sigma={args.sigma}) ...")
    new_sd, stats, untouched = transfer_weights(
        src_sd, moe.model, perturb_sigma=args.sigma, seed=args.seed
    )

    missing, unexpected = moe.model.load_state_dict(new_sd, strict=False)
    LOGGER.info(
        f"[init-moe-x] copied={stats['copied']}  "
        f"perturbed(routing experts)={stats['perturbed']}  "
        f"shape-mismatched={stats['mismatched']}  "
        f"kept-random={stats['kept_random']}  "
        f"missing={len(missing)}  unexpected={len(unexpected)}"
    )
    if args.verbose and untouched:
        LOGGER.info("[init-moe-x] keys kept at random init (truncated to 30):")
        for k in untouched[:30]:
            LOGGER.info(f"           {k}")

    # ---- save -------------------------------------------------------------
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    out = {
        "model": deepcopy(moe.model).half(),  # match ultralytics convention
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "best_fitness": None,
        "train_args": {"data": None, "imgsz": 640},
        "date": "",
        "version": "moe-final-conv-init-x",
    }
    torch.save(out, args.dst)
    LOGGER.info(f"[init-moe-x] saved -> {args.dst}")


if __name__ == "__main__":
    main()