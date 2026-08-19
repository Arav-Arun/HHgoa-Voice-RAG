"""Unified CLI — uv run hhgoa ingest | query | eval | bench | serve"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_ingest(args: argparse.Namespace) -> None:
    source = args.source
    if str(source).lower() == "msmarco":
        from eval.split import get_corpus_row_indices, load_split_config
        from ingest.indexer import ingest_msmarco_xi

        config = load_split_config()
        languages = tuple(args.languages)
        split = args.split or config.split
        if args.all:
            count = ingest_msmarco_xi(languages=languages, split=split, offset=0, limit=None)
            slice_label = "full split"
        else:
            corpus_indices = get_corpus_row_indices(config)
            count = ingest_msmarco_xi(
                languages=languages,
                split=split,
                row_indices=corpus_indices,
            )
            slice_label = (
                f"corpus shuffle[{config.corpus.shuffle_start}:"
                f"{config.corpus.shuffle_start + config.corpus.limit}), seed={config.shuffle_seed}"
            )
        lang_label = ", ".join(languages)
        print(
            f"Ingested {count} chunks from ai4bharat/MSMARCO-XI "
            f"({lang_label}, {split}, {slice_label})"
        )
        return

    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    from ingest.indexer import ingest_path

    count = ingest_path(source, language=args.language)
    print(f"Ingested {count} chunks from {source}")


def _cmd_query(args: argparse.Namespace) -> None:
    from core.factory import build_rag_pipeline

    pipeline = build_rag_pipeline(use_template_llm=args.template_llm)
    response = pipeline.query(args.question, language=args.language, top_k=args.top_k)

    print(f"\nQuery ({response.language}): {response.query}\n")
    print("Sources:")
    for i, source in enumerate(response.sources, start=1):
        preview = source.chunk.text[:120].replace("\n", " ")
        print(f"  {i}. [{source.score:.3f}] {preview}...")
    print("\nAnswer:\n")
    print(response.answer)


def _cmd_eval(args: argparse.Namespace) -> None:
    from eval.runner import run_eval

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")
    print(json.dumps(run_eval(args.eval_file, top_k=args.top_k), indent=2))


def _cmd_eval_build(args: argparse.Namespace) -> None:
    from eval.build import build_and_write_held_out_eval
    from eval.split import DEFAULT_SPLIT_PATH, EvalSplitConfig, load_split_config

    config = load_split_config(args.split_file)
    if tuple(args.languages) != config.languages:
        config = EvalSplitConfig(
            dataset=config.dataset,
            split=config.split,
            languages=tuple(args.languages),
            corpus=config.corpus,
            dev=config.dev,
            eval=config.eval,
        )
    examples = build_and_write_held_out_eval(
        args.output,
        split_path=args.split_file,
        config=config,
    )
    by_lang: dict[str, int] = {}
    for example in examples:
        by_lang[example.language] = by_lang.get(example.language, 0) + 1
    counts = ", ".join(f"{lang}={n}" for lang, n in sorted(by_lang.items())) or "none"
    spec = config.eval
    print(
        f"Wrote {len(examples)} held-out queries to {args.output} "
        f"({counts}; eval shuffle[{spec.shuffle_start}:{spec.shuffle_start + spec.limit}), "
        f"seed={config.shuffle_seed})"
    )
    print(f"Split config: {args.split_file}")


def _cmd_eval_validate(args: argparse.Namespace) -> None:
    from eval.validate import validate_eval_file

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")
    print(json.dumps(validate_eval_file(args.eval_file), indent=2))


def _cmd_bench(args: argparse.Namespace) -> None:
    from bench.runner import bench_query

    result = bench_query(args.query, runs=args.runs)
    print(
        json.dumps(
            {
                "operation": result.operation,
                "runs": result.runs,
                "mean_ms": round(result.mean_ms, 2),
                "p50_ms": round(result.p50_ms, 2),
                "p95_ms": round(result.p95_ms, 2),
            },
            indent=2,
        )
    )


def _cmd_eval_compare(args: argparse.Namespace) -> None:
    from eval.compare import compare_embedding_presets

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")

    results = compare_embedding_presets(
        tuple(args.presets),
        args.eval_file,
        top_k=args.top_k,
        ingest_limit=args.ingest_limit,
        ingest_split=args.split,
        baseline=args.baseline,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps(results, indent=2))


def _cmd_chunk_compare(args: argparse.Namespace) -> None:
    from eval.compare_chunking import compare_chunking_strategies

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")

    results = compare_chunking_strategies(
        tuple(args.strategies),
        args.eval_file,
        top_k=args.top_k,
        baseline=args.baseline,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps(results, indent=2))


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=args.reload,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hhgoa",
        description="Modular RAG framework (Hindi + Gujarati)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build vector index from documents")
    p_ingest.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=Path("msmarco"),
        help="Path to file/dir, or 'msmarco' for ai4bharat/MSMARCO-XI (hi + gu)",
    )
    p_ingest.add_argument("--language", choices=["hi", "gu"], default=None)
    p_ingest.add_argument(
        "--languages",
        nargs="+",
        choices=["hi", "gu"],
        default=["hi", "gu"],
        help="MS MARCO-XI languages to ingest (default: hi gu)",
    )
    p_ingest.add_argument(
        "--split",
        choices=["train", "validation"],
        default=None,
        help="MS MARCO-XI split (default: from data/eval/split.json)",
    )
    p_ingest.add_argument(
        "--all",
        action="store_true",
        help="Ingest the full msmarco split (ignore corpus slice in split.json)",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = sub.add_parser("query", help="Run a RAG query")
    p_query.add_argument("question")
    p_query.add_argument("--language", choices=["hi", "gu"], default="hi")
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.add_argument(
        "--template-llm",
        action="store_true",
        help="Show retrieved passages only (no LLM API call)",
    )
    p_query.set_defaults(func=_cmd_query)

    p_eval = sub.add_parser("eval", help="Run retrieval evaluation")
    p_eval.add_argument(
        "eval_file",
        type=Path,
        nargs="?",
        default=Path("data/eval/queries.jsonl"),
    )
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.set_defaults(func=_cmd_eval)

    p_eval_build = sub.add_parser(
        "eval-build",
        help="Build held-out data/eval/queries.jsonl + split.json from MSMARCO-XI",
    )
    p_eval_build.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/queries.jsonl"),
    )
    p_eval_build.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/eval/split.json"),
    )
    p_eval_build.add_argument(
        "--languages",
        nargs="+",
        choices=["hi", "gu"],
        default=["hi", "gu"],
    )
    p_eval_build.set_defaults(func=_cmd_eval_build)

    p_eval_validate = sub.add_parser(
        "eval-validate",
        help="Check held-out labels are present in the built index",
    )
    p_eval_validate.add_argument(
        "eval_file",
        type=Path,
        nargs="?",
        default=Path("data/eval/queries.jsonl"),
    )
    p_eval_validate.set_defaults(func=_cmd_eval_validate)

    p_eval_compare = sub.add_parser(
        "eval-compare",
        help="Re-ingest and compare embedding presets (per-language metrics)",
    )
    p_eval_compare.add_argument(
        "--presets",
        nargs="+",
        default=["e5-small", "indic-sbert", "bge-m3"],
        choices=["e5-small", "indic-sbert", "bge-m3"],
    )
    p_eval_compare.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/eval/queries.jsonl"),
    )
    p_eval_compare.add_argument("--top-k", type=int, default=5)
    p_eval_compare.add_argument(
        "--ingest-limit",
        type=int,
        default=500,
        help="MS MARCO-XI examples per language to index for each preset",
    )
    p_eval_compare.add_argument(
        "--baseline",
        default="e5-small",
        choices=["e5-small", "indic-sbert", "bge-m3"],
        help="Preset to compare others against in bootstrap tests",
    )
    p_eval_compare.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Bootstrap resamples for paired significance tests",
    )
    p_eval_compare.add_argument(
        "--split",
        choices=["train", "validation"],
        default="validation",
    )
    p_eval_compare.set_defaults(func=_cmd_eval_compare)

    p_chunk_compare = sub.add_parser(
        "chunk-compare",
        help="Re-ingest and compare chunking strategies on held-out eval",
    )
    p_chunk_compare.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "semantic", "metadata"],
        choices=["fixed", "semantic", "metadata"],
    )
    p_chunk_compare.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/eval/queries.jsonl"),
    )
    p_chunk_compare.add_argument("--top-k", type=int, default=5)
    p_chunk_compare.add_argument(
        "--baseline",
        default="fixed",
        choices=["fixed", "semantic", "metadata"],
    )
    p_chunk_compare.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
    )
    p_chunk_compare.set_defaults(func=_cmd_chunk_compare)

    p_bench = sub.add_parser("bench", help="Benchmark query latency")
    p_bench.add_argument("--query", default="भारत की राजधानी क्या है?")
    p_bench.add_argument("--runs", type=int, default=10)
    p_bench.set_defaults(func=_cmd_bench)

    p_serve = sub.add_parser("serve", help="Start HTTP API")
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
