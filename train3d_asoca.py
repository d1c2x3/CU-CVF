#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ASOCA training for SAM/CNN + CU-CVF + EUR.

Existing img/mask NPY files and folder*_train/validation.list are retained.
python train3d_asoca.py --data-path /path/to/npy --device cuda:0
python train3d_asoca.py --data-path /path/to/npy --resume auto
"""
import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


SCRIPT_DIR = Path(__file__).resolve().parent
# Defaults remain editable here, or can be overridden using JSON / CLI.
args = {
    "data_path": os.environ.get("ASOCA_DATA_PATH", "./data/ASOCA_dataset_2020/npy"),
    "split_root": str(SCRIPT_DIR / "dataset_ASOCA"), "folder": "folder5",
    "model_path": str(SCRIPT_DIR / "save_models_asoca_new"), "run_name": None,
    "sam_checkpoint": str(SCRIPT_DIR / "model/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth"),
    "input_shape": (128, 160, 160), "epochs": 800, "batch_size": 2,
    "lr": 1e-3, "sam_lr_ratio": 0.1, "weight_decay": 1e-5,
    "warmup_epochs": 10, "min_lr": 1e-6, "poly_power": 0.9,
    "class_weights": (0.2, 0.8), "lambda_ug": 1e-4, "kl_warmup_epochs": 50,
    "evidence_annealing_epochs": 50, "prior_gamma": 2.0, "prior_sigma0": 1.0,
    "prior_eps": 1e-6, "age_max_tokens": 1024, "sab_max_tokens": 512,
    "max_grad_norm": 5.0, "image_scale": 255.0, "patches_per_case": 1,
    "rotation_probability": 0.2, "rotation_degrees": 15.0,
    "flip_probability": 0.5, "blur_probability": 0.2, "blur_sigma": (0.1, 1.0),
    "num_workers": 4, "prefetch_factor": 2, "seed": 2026,
    "device": "auto", "amp": False, "deterministic": False, "cpu_threads": 4,
    "test_step": 1, "val_overlap": 0.5, "val_batch_size": 1,
    "threshold": 0.5, "keep_components": 2, "checkpoint_every": 10,
    "snapshot": 200, "log_every": 10, "resume": None, "initial_weights": None,
}


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config; explicit CLI settings take precedence")
    # Expose scalar defaults consistently without maintaining two default tables.
    for key, default in args.items():
        name = "--" + key.replace("_", "-")
        if key == "resume":
            parser.add_argument(name, nargs="?", const="auto", default=argparse.SUPPRESS)
        elif isinstance(default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif isinstance(default, (list, tuple)):
            parser.add_argument(name, nargs=len(default), type=type(default[0]), default=argparse.SUPPRESS)
        else:
            parser.add_argument(name, type=type(default) if default is not None else str, default=argparse.SUPPRESS)
    options = vars(parser.parse_args(argv))
    config = dict(args)
    filename = options.pop("config")
    if filename:
        supplied = json.loads(Path(filename).read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or set(supplied) - set(args):
            parser.error("JSON must be an object containing known configuration keys")
        config.update(supplied)
    config.update(options)
    return validate_config(config)


def validate_config(config):
    config = dict(config)
    positive_ints = ("epochs", "batch_size", "evidence_annealing_epochs", "cpu_threads", "test_step",
                     "checkpoint_every", "snapshot", "log_every", "patches_per_case", "val_batch_size", "prefetch_factor")
    nonnegative_ints = ("num_workers", "seed", "warmup_epochs", "kl_warmup_epochs", "keep_components")
    for key in positive_ints + nonnegative_ints:
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (1 if key in positive_ints else 0):
            raise ValueError(key + " must be a " + ("positive" if key in positive_ints else "nonnegative") + " integer")
    config["input_shape"] = tuple(config["input_shape"])
    if len(config["input_shape"]) != 3 or any(not isinstance(n, int) or n < 16 or n % 16 for n in config["input_shape"]):
        raise ValueError("input_shape must contain three positive multiples of 16")
    for key in ("lr", "min_lr", "sam_lr_ratio", "poly_power", "image_scale", "prior_sigma0", "prior_eps", "max_grad_norm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    for key in ("lambda_ug", "prior_gamma", "weight_decay", "rotation_degrees"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(key + " must be finite and nonnegative")
    for key in ("rotation_probability", "flip_probability", "blur_probability", "threshold"):
        if not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(key + " must lie in [0,1]")
    if not 0 <= config["val_overlap"] < 1 or config["min_lr"] > config["lr"]:
        raise ValueError("val_overlap must lie in [0,1); min_lr must not exceed lr")
    if len(config["blur_sigma"]) != 2 or not 0 < config["blur_sigma"][0] <= config["blur_sigma"][1]:
        raise ValueError("blur_sigma must be a positive ordered pair")
    weights = np.asarray(config["class_weights"], dtype=float)
    if weights.shape != (2,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("class_weights must be two finite positive values")
    for key in ("age_max_tokens", "sab_max_tokens"):
        if config[key] is not None and (not isinstance(config[key], int) or config[key] < 1):
            raise ValueError(key + " must be a positive integer or null")
    if config["resume"] and config["initial_weights"]:
        raise ValueError("Choose resume or initial_weights")
    return config


def load_case_pairs(config, split):
    listing = Path(config["split_root"]).expanduser().resolve() / config["folder"] / (config["folder"] + "_" + split + ".list")
    names = [line.strip() for line in listing.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Split must be nonempty with unique filenames: " + str(listing))
    root = Path(config["data_path"]).expanduser().resolve()
    pairs = [(root / "img" / name, root / "mask" / name) for name in names]
    for image, label in pairs:
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError("Missing image/mask pair: " + str(image) + " ; " + str(label))
        a, b = np.load(image, mmap_mode="r", allow_pickle=False), np.load(label, mmap_mode="r", allow_pickle=False)
        if a.ndim != 3 or a.shape != b.shape or min(a.shape) < 1:
            raise ValueError("Expected matching, nonempty 3D image/mask arrays: " + str(image))
    return pairs


def split_manifest(train_pairs, val_pairs):
    if set(p[0].stem for p in train_pairs) & set(p[0].stem for p in val_pairs):
        raise ValueError("Training and validation cases overlap")
    manifest = {name: [{"case": a.name, "image": str(a), "mask": str(b),
                        "shape": list(np.load(a, mmap_mode="r", allow_pickle=False).shape)} for a, b in pairs]
                for name, pairs in (("train", train_pairs), ("validation", val_pairs))}
    return manifest, hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


class ASOCAPatchDataset(Dataset):
    """Uniform crops and seed/epoch/index-based augmentation, independent of worker RNG.

    Small arrays are padded; masks use nearest-neighbor rotation and are never
    resized with image interpolation. Existing 0/255 and 0/1 masks both work.
    """
    def __init__(self, pairs, config):
        self.pairs, self.config = pairs, config

    def __len__(self):
        return len(self.pairs) * self.config["patches_per_case"]

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) else (0, item)
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], epoch, index]))
        image_path, label_path = self.pairs[index % len(self.pairs)]
        image, label = np.load(image_path, mmap_mode="r", allow_pickle=False), np.load(label_path, mmap_mode="r", allow_pickle=False)
        patch = self.config["input_shape"]
        starts = [int(rng.integers(max(0, n - p) + 1)) for n, p in zip(image.shape, patch)]
        roi = tuple(slice(s, s + p) for s, p in zip(starts, patch))
        image = np.asarray(image[roi], dtype=np.float32) / self.config["image_scale"]
        label = (np.asarray(label[roi]) > 0).astype(np.uint8)
        padding = [(0, max(0, p - n)) for p, n in zip(patch, image.shape)]
        image, label = np.pad(image, padding), np.pad(label, padding)
        if not np.isfinite(image).all():
            raise ValueError("Nonfinite image values: " + str(image_path))
        if rng.random() < self.config["rotation_probability"]:
            angle = rng.uniform(-self.config["rotation_degrees"], self.config["rotation_degrees"])
            image = ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=1, mode="constant", prefilter=False)
            label = ndimage.rotate(label, angle, axes=(1, 2), reshape=False, order=0, mode="constant", prefilter=False)
        for axis in range(3):
            if rng.random() < self.config["flip_probability"]:
                image, label = np.flip(image, axis), np.flip(label, axis)
        if rng.random() < self.config["blur_probability"]:
            image = ndimage.gaussian_filter(image, sigma=rng.uniform(*self.config["blur_sigma"]))
        return (torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(label[None], dtype=np.int64)))


class EpochRandomSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter((self.epoch, i) for i in torch.randperm(len(self.dataset), generator=generator).tolist())


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def setup_runtime(config):
    name = config["device"]
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if name == "auto" else name)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("device must be auto, cpu, or cuda:N")
    if config["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    if config["amp"] and device.type != "cuda":
        raise ValueError("AMP requires CUDA here; use --no-amp on CPU")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = not config["deterministic"]
    torch.backends.cudnn.deterministic = config["deterministic"]
    torch.use_deterministic_algorithms(config["deterministic"])
    return device


def build_optimizer(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        sam = name.startswith("sam_model.")
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        key = (sam, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": config["lr"] * (config["sam_lr_ratio"] if sam else 1.0),
                           "weight_decay": config["weight_decay"] if decay else 0.0,
                           "name": ("sam" if sam else "new") + ("_decay" if decay else "_no_decay")}
        groups[key]["params"].append(parameter)
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.Adam(list(groups.values()))


def learning_rate_factor(epoch, config):
    warmup = min(config["warmup_epochs"], max(0, config["epochs"] - 1))
    if epoch < warmup:
        return float(epoch + 1) / warmup
    progress = min(1.0, max(0.0, (epoch - warmup) / max(1, config["epochs"] - warmup)))
    return max(config["min_lr"] / config["lr"], (1.0 - progress) ** config["poly_power"])


def effective_kl_weight(epoch, config):
    ramp = config["kl_warmup_epochs"]
    return config["lambda_ug"] * (min(1.0, epoch / ramp) if ramp else 1.0)


def metrics_from_counts(tp, fp, fn, tn):
    return {"dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "sensitivity": tp / (tp + fn) if tp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else float(fn == 0),
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def binary_metrics(prediction, target):
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    return metrics_from_counts(int(np.count_nonzero(pred & truth)), int(np.count_nonzero(pred & ~truth)),
                               int(np.count_nonzero(~pred & truth)), int(np.count_nonzero(~pred & ~truth)))


def sliding_window_rois(shape, patch, overlap):
    starts = []
    for length, size in zip(shape, patch):
        if length < size:
            raise ValueError("Pad small volumes before requesting windows")
        stride = max(1, int(size * (1.0 - overlap)))
        axis = list(range(0, length - size + 1, stride))
        if axis[-1] != length - size:
            axis.append(length - size)
        starts.append(axis)
    return itertools.product(*starts)


def get_patch_new(img_shape, interval_0=128, interval_1=160, step=2):
    if step < 1:
        raise ValueError("step must be at least one")
    starts = []
    for length, patch in zip(img_shape, (interval_0, interval_1, interval_1)):
        if length < patch:
            raise ValueError("Pad small volumes before requesting windows")
        stride = max(1, int(patch / step))
        axis = list(range(0, length - patch + 1, stride))
        if axis[-1] != length - patch:
            axis.append(length - patch)
        starts.append(axis)
    return [[z, y, x] for z in starts[0] for y in starts[1] for x in starts[2]]


@torch.inference_mode()
def predict_volume(model, image, config, device):
    shape, patch = image.shape, tuple(config["input_shape"])
    padding = [(0, max(0, p - n)) for p, n in zip(patch, shape)]
    image = np.pad(np.asarray(image, dtype=np.float32), padding) / config["image_scale"]
    if not np.isfinite(image).all():
        raise ValueError("Validation image contains nonfinite values")
    probability_sum, count = np.zeros(image.shape, dtype=np.float32), np.zeros(image.shape, dtype=np.float32)
    windows = iter(sliding_window_rois(image.shape, patch, config["val_overlap"]))
    while True:
        positions = list(itertools.islice(windows, config["val_batch_size"]))
        if not positions:
            break
        slices = [tuple(slice(s, s + p) for s, p in zip(pos, patch)) for pos in positions]
        batch = torch.from_numpy(np.stack([image[roi] for roi in slices])[:, None]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config["amp"]):
            logits = model(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite validation logits")
        probability = logits.float().softmax(1)[:, 1].cpu().numpy()
        for roi, prob in zip(slices, probability):
            probability_sum[roi] += prob
            count[roi] += 1.0
    if (count == 0).any():
        raise RuntimeError("Sliding windows left uncovered voxels")
    averaged = probability_sum / count
    return averaged[tuple(slice(0, n) for n in shape)]


def keep_largest_components(mask, number=2):
    if number <= 0:
        return mask
    components, count = ndimage.label(mask)
    if count <= number:
        return mask
    sizes = np.bincount(components.ravel())
    return np.isin(components, np.argsort(sizes[1:])[-number:] + 1)


def model_eval_patch(net, config=None, pairs=None, device=None):
    config = dict(args) if config is None else config
    pairs = load_case_pairs(config, "validation") if pairs is None else pairs
    device = next(net.parameters()).device if device is None else device
    was_training = net.training
    net.eval()
    rows = []
    try:
        for image_path, label_path in pairs:
            image = np.load(image_path, mmap_mode="r", allow_pickle=False)
            truth = np.load(label_path, mmap_mode="r", allow_pickle=False) > 0
            probability = predict_volume(net, image, config, device)
            pred = keep_largest_components(probability >= config["threshold"], config["keep_components"])
            rows.append({"case": image_path.name, **binary_metrics(pred, truth)})
    finally:
        net.train(was_training)
    if not rows:
        raise ValueError("Validation set is empty")
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "case"}, rows


def train_epoch(model, loader, optimizer, scaler, criterion, config, epoch, device, logger):
    model.train()
    criterion.lambda_ug = effective_kl_weight(epoch, config)
    fields = ("total", "segmentation", "dice", "weighted_ce", "evidence", "evidence_kl", "ug_kl")
    totals = torch.zeros(len(fields) + 2, device=device, dtype=torch.float64)
    counts = torch.zeros(4, device=device, dtype=torch.int64)
    samples, start = 0, time.perf_counter()
    for step, (image, label) in enumerate(loader, 1):
        image = image.to(device, dtype=torch.float32, non_blocking=True)
        label = label.to(device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config["amp"]):
            output = model(image, return_aux=True)
        # Dirichlet special functions and the loss use FP32 even with optional AMP.
        losses = criterion(output, label, epoch=epoch)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("Nonfinite loss at epoch {}, step {}".format(epoch + 1, step))
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config["max_grad_norm"], error_if_nonfinite=not config["amp"])
        scaler.step(optimizer)
        scaler.update()
        batch_size = image.shape[0]
        with torch.no_grad():
            values = torch.stack([losses[k].detach() for k in fields] +
                                 [output["uncertainty"].detach().mean(), grad_norm.detach()])
            totals += values.double() * batch_size
            pred, truth = output["logits"].argmax(1).bool(), label[:, 0].bool()
            counts += torch.stack(((pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum(), (~pred & ~truth).sum()))
        samples += batch_size
        if step == 1 or step % config["log_every"] == 0 or step == len(loader):
            average = (totals[:len(fields)] / samples).cpu().tolist()
            logger.info("epoch=%d/%d step=%d/%d loss=%.5f seg=%.5f evidence=%.5f KL=%.5f weighted_KL=%.5f",
                        epoch + 1, config["epochs"], step, len(loader), average[0], average[1], average[4],
                        average[6], criterion.lambda_ug * average[6])
        del output, losses, image, label
    if samples == 0:
        raise RuntimeError("Training loader produced no batches")
    result = dict(zip(fields + ("uncertainty", "grad_norm"), (totals / samples).cpu().tolist()))
    result["dice_loss"] = result.pop("dice")
    result.update(metrics_from_counts(*counts.cpu().tolist()))
    result.update(lambda_ug=criterion.lambda_ug, weighted_ug_kl=criterion.lambda_ug * result["ug_kl"], seconds=time.perf_counter() - start)
    return result


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_csv(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def capture_rng(device, generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "loader": generator.get_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def resume_checkpoint(path, model, optimizer, scheduler, scaler, config, signature, device, generator):
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "scaler", "config", "rng", "epoch", "data_signature",
                "format_version", "global_step", "best_dice", "best_epoch"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("Resume needs a full training checkpoint; use --initial-weights for a state_dict")
    if state["format_version"] != 1:
        raise ValueError("Unsupported training checkpoint format")
    runtime = {"data_path", "split_root", "model_path", "run_name", "sam_checkpoint", "num_workers", "prefetch_factor",
               "device", "cpu_threads", "checkpoint_every", "snapshot", "log_every", "resume", "initial_weights"}
    for key in set(config) - runtime:
        if json.dumps(state["config"][key], sort_keys=True) != json.dumps(config[key], sort_keys=True):
            raise ValueError("Resume setting differs: " + key + "; use initial_weights for a new training trajectory")
    if signature != state["data_signature"]:
        raise ValueError("Dataset paths/case lists/shapes differ from the resumed run")
    model.load_state_dict(state["model"], strict=True)
    scheduler.load_state_dict(state["scheduler"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"], device, generator)
    return state


def train(config=None, model=None):
    """model injection permits offline end-to-end tests with a small SAM encoder."""
    config = validate_config(dict(args) if config is None else config)
    device = setup_runtime(config)
    training, validation = load_case_pairs(config, "train"), load_case_pairs(config, "validation")
    manifest, signature = split_manifest(training, validation)
    name = config["run_name"] or "CSNet3D_sam_{}_b{}_{}".format(config["folder"], config["batch_size"], "x".join(map(str, config["input_shape"])))
    run_dir = Path(config["model_path"]).expanduser().resolve() / name
    resume = config["resume"]
    resume_path = run_dir / "last_checkpoint.pt" if resume == "auto" else (Path(resume).expanduser().resolve() if resume else None)
    if resume_path and not resume_path.is_file():
        raise FileNotFoundError("Resume checkpoint not found: " + str(resume_path))
    if not resume and (run_dir / "config.json").exists():
        raise FileExistsError("Run exists; use --resume auto or another --run-name: " + str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("asoca." + str(run_dir))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        from model.CUCVF3D_sam import CUCVF3D_sam, CUCVFCompositeLoss
        if model is None:
            model = CUCVF3D_sam(2, 1, ckpt_path=config["sam_checkpoint"], prior_gamma=config["prior_gamma"],
                               prior_sigma0=config["prior_sigma0"], prior_eps=config["prior_eps"],
                               age_max_tokens=config["age_max_tokens"], sab_max_tokens=config["sab_max_tokens"])
        model = model.to(device)
        if config["initial_weights"]:
            model.load_state_dict(torch.load(config["initial_weights"], map_location="cpu", weights_only=True), strict=True)
        criterion = CUCVFCompositeLoss(config["class_weights"], config["lambda_ug"], config["evidence_annealing_epochs"]).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: learning_rate_factor(epoch, config))
        scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
        dataset = ASOCAPatchDataset(training, config)
        sampler = EpochRandomSampler(dataset, config["seed"])
        generator = torch.Generator().manual_seed(config["seed"])
        loader_options = {"dataset": dataset, "batch_size": config["batch_size"], "sampler": sampler,
                          "num_workers": config["num_workers"], "pin_memory": device.type == "cuda",
                          "worker_init_fn": seed_worker, "generator": generator, "drop_last": False}
        if config["num_workers"]:
            loader_options.update(persistent_workers=True, prefetch_factor=config["prefetch_factor"])
        loader = DataLoader(**loader_options)
        start_epoch, global_step, best_dice, best_epoch = 0, 0, -1.0, 0
        if resume_path:
            restored = resume_checkpoint(resume_path, model, optimizer, scheduler, scaler, config, signature, device, generator)
            start_epoch, global_step = restored["epoch"], restored["global_step"]
            best_dice, best_epoch = restored["best_dice"], restored["best_epoch"]
            logger.info("Resumed completed epoch %d; best DSC %.6f at epoch %d", start_epoch, best_dice, best_epoch)
        (run_dir / ("config.resume.json" if resume_path else "config.json")).write_text(json.dumps(config, indent=2), encoding="utf-8")
        (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("device=%s train_cases=%d val_cases=%d parameters=%d trainable=%d amp=%s", device, len(training), len(validation),
                    sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad), config["amp"])
        logger.info("LR=%g SAM_LR=%g weights=%s UG-KL=%g ramp=%d evidence_anneal=%d", config["lr"], config["lr"] * config["sam_lr_ratio"],
                    config["class_weights"], config["lambda_ug"], config["kl_warmup_epochs"], config["evidence_annealing_epochs"])
        for epoch in range(start_epoch, config["epochs"]):
            sampler.epoch = epoch
            used_lr = {group["name"]: group["lr"] for group in optimizer.param_groups}
            metrics = train_epoch(model, loader, optimizer, scaler, criterion, config, epoch, device, logger)
            global_step += len(loader)
            val = {key: "" for key in ("dice", "iou", "sensitivity", "precision", "fpr")}
            if (epoch + 1) % config["test_step"] == 0 or epoch + 1 == config["epochs"]:
                val, cases = model_eval_patch(model, config, validation, device)
                for case in cases:
                    append_csv(run_dir / "validation_cases.csv", {"epoch": epoch + 1, **case})
                logger.info("Validation epoch=%d DSC=%.6f IoU=%.6f SE=%.6f precision=%.6f FPR=%.6f", epoch + 1,
                            *[val[k] for k in ("dice", "iou", "sensitivity", "precision", "fpr")])
                if val["dice"] > best_dice:
                    best_dice, best_epoch = val["dice"], epoch + 1
                    atomic_torch_save(model.state_dict(), run_dir / "best_score_checkpoint.pkl")
                    (run_dir / "best_metrics.json").write_text(json.dumps({"epoch": best_epoch, **val}, indent=2), encoding="utf-8")
            scheduler.step()
            append_csv(run_dir / "metrics.csv", {"epoch": epoch + 1, "global_step": global_step,
                       **{"lr_" + k: v for k, v in used_lr.items()}, **{"train_" + k: v for k, v in metrics.items()},
                       **{"val_" + k: v for k, v in val.items()}, "best_dice": best_dice})
            if epoch == 0 or (epoch + 1) % config["checkpoint_every"] == 0 or epoch + 1 == config["epochs"] or (epoch + 1) % config["snapshot"] == 0:
                checkpoint = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1,
                              "global_step": global_step, "best_dice": best_dice, "best_epoch": best_epoch,
                              "config": config, "data_signature": signature, "rng": capture_rng(device, generator)}
                atomic_torch_save(checkpoint, run_dir / "last_checkpoint.pt")
                if (epoch + 1) % config["snapshot"] == 0:
                    atomic_torch_save(checkpoint, run_dir / "epoch_{:04d}.pt".format(epoch + 1))
            logger.info("Finished epoch=%d time=%.1fs train_DSC=%.6f best_DSC=%.6f best_epoch=%d",
                        epoch + 1, metrics["seconds"], metrics["dice"], best_dice, best_epoch)
        return {"run_dir": str(run_dir), "best_dice": best_dice, "best_epoch": best_epoch,
                "completed_epochs": max(start_epoch, config["epochs"])}
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    train(parse_config())
