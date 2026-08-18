"""Unified CLI — uv run hhgoa ingest | query | eval | bench | serve"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_ingest(args: argparse.Namespace) -> None:
    from ingest.indexer import ingest_path

    source = args.source
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
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
        default=Path("data/samples"),
    )
    p_ingest.add_argument("--language", choices=["hi", "gu"], default=None)
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
