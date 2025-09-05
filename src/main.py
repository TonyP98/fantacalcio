"""Command line interface for fantacalcio."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import dataio, features, pricing, ranking, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fantacalcio CLI")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Import player data")
    p_ingest.add_argument("--input", required=True)

    sub.add_parser("build-features", help="Generate features")

    p_price = sub.add_parser("price", help="Estimate prices")
    p_price.add_argument("--method", choices=["baseline", "heuristic"], default="baseline")

    p_rank = sub.add_parser("rank", help="Rank players")
    p_rank.add_argument("--by", default="value")
    p_rank.add_argument("--role", default="ALL")
    p_rank.add_argument("--top", type=int, default=10)
    p_rank.add_argument("--budget", type=float, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = utils.load_config(Path(args.config))
    seed = args.seed or config.get("seed", 42)
    utils.set_seed(seed)
    logger = utils.get_logger(args.verbose)

    if args.command == "ingest":
        df = dataio.load_csv(Path(args.input), config)
        out_path = utils.resolve_path(config, "interim") / "players.parquet"
        dataio.save_parquet(df, out_path)
        logger.info("Saved interim data to %s", out_path)
    elif args.command == "build-features":
        in_path = utils.resolve_path(config, "interim") / "players.parquet"
        df = dataio.load_parquet(in_path)
        feats = features.build_features(df)
        out_path = utils.resolve_path(config, "processed") / "features.parquet"
        dataio.save_parquet(feats, out_path)
        logger.info("Saved features to %s", out_path)
    elif args.command == "price":
        in_path = utils.resolve_path(config, "processed") / "features.parquet"
        df = dataio.load_parquet(in_path)
        if args.method == "baseline":
            priced = pricing.baseline_linear(df, config["budget"])
        else:
            priced = pricing.heuristic_price(df, config["scoring_weights"], config["budget"])
        out_path = utils.resolve_path(config, "processed") / "prices.parquet"
        dataio.save_parquet(priced, out_path)
        logger.info("Saved prices to %s", out_path)
    elif args.command == "rank":
        in_path = utils.resolve_path(config, "processed") / "prices.parquet"
        df = dataio.load_parquet(in_path)
        ranked = ranking.rank_players(df, args.by, args.role, args.top, args.budget)
        out_path = utils.resolve_path(config, "outputs") / "ranking.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out_path, index=False)
        logger.info("Saved ranking to %s", out_path)


if __name__ == "__main__":
    main()
