"""Command-line interface for the semantic-only reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .attacks import ProgressiveGradientAscent
from .channel import AWGNChannel
from .config import load_config
from .data import _load_array, normalize_csi, prepare_cost2100_array
from .metrics import distortion_per_sample, target_distortion
from .model import DeepJSCC
from .runtime import evaluate_attacks, evaluate_clean, plot_results, train
from .theory import (
    clean_distortion_upper_bound,
    lemma1_attack_power_lower_bound,
    theorem3_attack_power_lower_bound,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic-only reproduction of Zhang et al. (2026)."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train an image or CSI DeepJSCC model.")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output")
    train_parser.add_argument("--device", default="auto")

    clean_parser = commands.add_parser("clean", help="Evaluate the no-attack SNR curve.")
    clean_parser.add_argument("--config", required=True)
    clean_parser.add_argument("--checkpoint", required=True)
    clean_parser.add_argument("--output")
    clean_parser.add_argument("--device", default="auto")

    attack_parser = commands.add_parser("attack", help="Evaluate PGA and/or C&W.")
    attack_parser.add_argument("--config", required=True)
    attack_parser.add_argument("--checkpoint", required=True)
    attack_parser.add_argument("--attacks", nargs="+", choices=["pga", "cw"], default=["pga", "cw"])
    attack_parser.add_argument("--output")
    attack_parser.add_argument("--device", default="auto")

    plot_parser = commands.add_parser("plot", help="Plot semantic clean/attack curves.")
    plot_parser.add_argument("--config", required=True)
    plot_parser.add_argument("--clean-csv")
    plot_parser.add_argument("--attack-csv")
    plot_parser.add_argument("--output", required=True)

    prepare_parser = commands.add_parser(
        "prepare-csi", help="Convert raw COST2100 CSI into normalized 32x32 arrays."
    )
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--key")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument(
        "--representation",
        choices=["magnitude", "real", "imaginary", "real_imag"],
        default="magnitude",
    )
    prepare_parser.add_argument("--already-angular-delay", action="store_true")
    prepare_parser.add_argument("--normalization-min", type=float)
    prepare_parser.add_argument("--normalization-max", type=float)

    theory_parser = commands.add_parser("theory", help="Evaluate semantic bounds in Eqs. (10)-(12).")
    theory_parser.add_argument("--target-distortion", type=float, required=True)
    theory_parser.add_argument("--clean-distortion", type=float)
    theory_parser.add_argument("--lipschitz", type=float, required=True)
    theory_parser.add_argument("--channel-uses", type=int, required=True)
    theory_parser.add_argument("--noise-variance", type=float, required=True)

    smoke_parser = commands.add_parser("smoke", help="Run a synthetic end-to-end gradient smoke test.")
    smoke_parser.add_argument("--task", choices=["image", "csi"], default="image")
    smoke_parser.add_argument("--device", default="cpu")
    return parser


def _prepare_csi(args: argparse.Namespace) -> None:
    values = _load_array(Path(args.input), args.key)
    prepared = prepare_cost2100_array(
        values,
        representation=args.representation,
        already_angular_delay=args.already_angular_delay,
    )
    normalized, minimum, maximum = normalize_csi(
        prepared, minimum=args.normalization_min, maximum=args.normalization_max
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".npy":
        np.save(destination, normalized)
    else:
        np.savez_compressed(
            destination,
            csi=normalized,
            normalization_min=np.asarray(minimum),
            normalization_max=np.asarray(maximum),
            representation=np.asarray(args.representation),
        )
    print(
        json.dumps(
            {
                "output": str(destination),
                "shape": list(normalized.shape),
                "minimum": minimum,
                "maximum": maximum,
                "representation": args.representation,
            },
            indent=2,
        )
    )


def _theory(args: argparse.Namespace) -> None:
    result = {
        "eq11_clean_distortion_upper_bound": clean_distortion_upper_bound(
            args.lipschitz, args.channel_uses, args.noise_variance
        ),
        "eq12_attack_power_lower_bound": theorem3_attack_power_lower_bound(
            args.target_distortion,
            args.lipschitz,
            args.channel_uses,
            args.noise_variance,
        ),
    }
    if args.clean_distortion is not None:
        result["eq10_attack_power_lower_bound"] = lemma1_attack_power_lower_bound(
            args.target_distortion, args.clean_distortion, args.lipschitz
        )
    print(json.dumps(result, indent=2))


def _smoke(task: str, device_name: str) -> None:
    torch.manual_seed(2026)
    device = torch.device(device_name)
    channels = 3 if task == "image" else 1
    multiplier = 6 if task == "image" else 2
    model = DeepJSCC(channels, multiplier).to(device)
    channel = AWGNChannel().to(device)
    inputs = torch.rand(2, channels, 32, 32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    reconstruction = model(inputs, channel, 10.0)
    loss = torch.nn.functional.mse_loss(reconstruction, inputs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        symbols = model.encode(inputs)
        received = channel(symbols, 10.0)
    threshold = target_distortion(task, 15.0 if task == "image" else -16.0)
    result = ProgressiveGradientAscent(step_size=0.1, max_steps=2)(
        model.decoder,
        inputs,
        received,
        lambda target, decoded: distortion_per_sample(task, target, decoded),
        threshold,
    )
    print(
        json.dumps(
            {
                "task": task,
                "channel_uses": model.channel_uses,
                "bandwidth_ratio": model.bandwidth_ratio,
                "training_loss": float(loss.detach()),
                "pga_distortion": result.distortion.cpu().tolist(),
                "pga_success": result.success.cpu().tolist(),
                "status": "ok",
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train":
        path = train(load_config(args.config), output_override=args.output, device_name=args.device)
        print(path)
    elif args.command == "clean":
        path = evaluate_clean(
            load_config(args.config),
            args.checkpoint,
            output_override=args.output,
            device_name=args.device,
        )
        print(path)
    elif args.command == "attack":
        paths = evaluate_attacks(
            load_config(args.config),
            args.checkpoint,
            args.attacks,
            output_override=args.output,
            device_name=args.device,
        )
        print("\n".join(map(str, paths)))
    elif args.command == "plot":
        path = plot_results(
            load_config(args.config), args.clean_csv, args.attack_csv, args.output
        )
        print(path)
    elif args.command == "prepare-csi":
        _prepare_csi(args)
    elif args.command == "theory":
        _theory(args)
    elif args.command == "smoke":
        _smoke(args.task, args.device)


if __name__ == "__main__":
    main()
