"""Unified CLI, uv run hhgoa ingest | query | eval | bench | serve"""

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


def _render_response(response) -> None:
    """Shared rendering for `query` and `voice-query`."""
    print(f"\nQuery ({response.language}): {response.query}\n")
    print("Sources:")
    for i, source in enumerate(response.sources, start=1):
        preview = source.chunk.text[:120].replace("\n", " ")
        parts = source.components
        detail = ""
        if "sparse_score" in parts:
            detail = f" (dense={parts.get('dense_score', 0):.3f} bm25={parts.get('sparse_score', 0):.2f})"
        print(f"  {i}. [{source.score:.4f}]{detail} {preview}...")

    print("\nAnswer:\n")
    print(response.answer or "(abstained)")

    meta = response.metadata
    guardrail = meta.get("guardrail")
    if guardrail:
        print(f"\n[guardrail] stage={guardrail.get('stage')} reason={guardrail.get('reason')}")
    quality = meta.get("quality")
    if quality:
        print(f"[quality] {json.dumps(quality, ensure_ascii=False)}")
    timings = meta.get("timings_ms", {})
    if timings:
        breakdown = "  ".join(f"{k}={v:.1f}ms" for k, v in timings.items())
        print(f"\n[latency] total={meta.get('total_ms', 0):.1f}ms  path={meta.get('path')}")
        print(f"[stages]  {breakdown}")


def _cmd_query(args: argparse.Namespace) -> None:
    from core.factory import build_rag_pipeline

    pipeline = build_rag_pipeline(local_only=args.local_only)
    response = pipeline.query(
        args.question,
        language=args.language,
        top_k=args.top_k,
        mode=args.mode,
    )
    _render_response(response)


def _cmd_eval(args: argparse.Namespace) -> None:
    from eval.runner import run_eval

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")
    print(json.dumps(run_eval(args.eval_file, top_k=args.top_k), indent=2))


def _cmd_eval_build(args: argparse.Namespace) -> None:
    from eval.build import build_and_write_dev_eval, build_and_write_held_out_eval
    from eval.split import EvalSplitConfig, load_split_config

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
    # The dev slice comes from the same split config and is what every sweep
    # tunes on, so building one without the other leaves a dangling dependency.
    dev = build_and_write_dev_eval(args.dev_output, config=config)
    dev_spec = config.dev
    print(
        f"Wrote {len(dev)} dev queries to {args.dev_output} "
        f"(dev shuffle[{dev_spec.shuffle_start}:{dev_spec.shuffle_start + dev_spec.limit}))"
    )
    print(f"Split config: {args.split_file}")


def _cmd_eval_validate(args: argparse.Namespace) -> None:
    from eval.validate import validate_eval_file

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")
    print(json.dumps(validate_eval_file(args.eval_file), indent=2))


def _cmd_transcribe(args: argparse.Namespace) -> None:
    from core.factory import build_stt

    audio_path = args.audio
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    audio_bytes = audio_path.read_bytes()
    content_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(audio_path.suffix.lower(), "application/octet-stream")

    try:
        result = build_stt().transcribe(
            audio_bytes,
            language=args.language,
            content_type=content_type,
            filename=audio_path.name,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps({"text": result.text, "language": result.language, "provider": result.provider}, ensure_ascii=False, indent=2))


def _cmd_voice_query(args: argparse.Namespace) -> None:
    from core.factory import build_rag_pipeline, build_stt

    audio_path = args.audio
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    audio_bytes = audio_path.read_bytes()
    content_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(audio_path.suffix.lower(), "application/octet-stream")

    try:
        transcription = build_stt().transcribe(
            audio_bytes,
            language=args.language,
            content_type=content_type,
            filename=audio_path.name,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    pipeline = build_rag_pipeline(local_only=args.local_only)
    response = pipeline.query(
        transcription.text,
        language=args.language,
        top_k=args.top_k,
        mode=args.mode,
    )

    print(f"\nTranscription ({args.language}): {transcription.text}")
    _render_response(response)


def _cmd_guardrail_calibrate(args: argparse.Namespace) -> None:
    from eval.calibrate_guardrail import calibrate_and_write

    report = calibrate_and_write(
        answerable_path=args.answerable_file,
        abstain_path=args.abstain_file,
        output_path=args.output,
        top_k=args.top_k,
    )
    default = report["shipping_default"]["held_out"]
    base = report["threshold_baseline"]["held_out"]
    gate = report["multi_feature_gate"]["held_out"]
    print(f"samples: {report['samples']['total']}  {report['samples']['by_category']}")
    print(f"split:   fit={report['split']['fit']} held_out={report['split']['held_out']}\n")
    default_label = f"cosine {report['shipping_default']['threshold']:.3f}"
    swept_label = f"swept {report['threshold_baseline']['threshold']:.3f}"
    print(f"{'metric':24} {default_label:>15} {swept_label:>15} {'multi-feature':>15}")
    print("-" * 74)
    for key in (
        "answerable_recall",
        "false_abstain_rate",
        "abstain_recall",
        "balanced_accuracy",
        "answerable_f1",
    ):
        print(f"{key:24} {default[key]:15.4f} {base[key]:15.4f} {gate[key]:15.4f}")
    rep = report["multi_feature_gate"].get("repeated_holdout")
    if rep:
        print(
            f"\nover {int(rep['splits'])} random half-splits (what the docs quote, since one\n"
            f"split leaves ~48 abstain examples and swings by several points):\n"
            f"  answerable_recall {rep['answerable_recall']:.4f}   "
            f"false_abstain {rep['false_abstain_rate']:.4f}   "
            f"abstain_recall {rep['abstain_recall']:.4f}   "
            f"balanced_accuracy {rep['balanced_accuracy']:.4f} +/- {rep['balanced_accuracy_stdev']:.4f}"
        )
    print(f"\ncoefficients: {report['multi_feature_gate']['coefficients']}")
    print(f"operating threshold: {report['multi_feature_gate']['threshold']}")
    print(f"wrote {args.output} and {report['model_path']}")


def _cmd_bench(args: argparse.Namespace) -> None:
    from bench.runner import run_benchmarks

    report = run_benchmarks(
        limit=args.queries,
        modes=tuple(args.modes),
        audio_dir=args.audio_dir,
        output_path=args.output,
    )

    for mode, track in report["tracks"].items():
        if mode == "voice":
            print(f"\n=== voice end-to-end ({track['clips']} clips) ===")
            for name in ("stt", "rag", "end_to_end"):
                st = track[name]
                print(f"  {name:11} p50={st['p50_ms']:8.1f}  p70={st['p70_ms']:8.1f}  p100={st['p100_ms']:8.1f}")
            continue

        total = track["total"]
        print(f"\n=== {mode} path, {track['queries']} queries ===")
        print(
            f"  TOTAL       p50={total['p50_ms']:8.2f}  p70={total['p70_ms']:8.2f}  "
            f"p100={total['p100_ms']:8.2f}   (mean={total['mean_ms']:.2f})"
        )
        print(f"  cold start  {track['cold_start_ms']:.1f} ms (excluded from percentiles)")
        print("  per stage:")
        for stage, st in track["by_stage"].items():
            print(
                f"    {stage:22} p50={st['p50_ms']:7.2f}  p70={st['p70_ms']:7.2f}  p100={st['p100_ms']:7.2f}"
            )
        target = report["target_ms"]
        verdict = "PASS" if total["p100_ms"] < target else "FAIL"
        print(f"  <{target}ms at P100: {verdict}")

    if args.output:
        print(f"\nWrote {args.output}")


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


def _cmd_child_sweep(args: argparse.Namespace) -> None:
    from eval.sweep_child_size import sweep_child_size

    for path in (args.eval_file, args.dev_file):
        if not path.exists():
            raise SystemExit(f"Query file not found: {path}")

    result = sweep_child_size(
        args.eval_file,
        args.dev_file,
        sizes=tuple(args.sizes),
        top_k=args.top_k,
        reuse_index=args.reuse_index,
        output_path=args.output,
    )
    k = args.top_k
    hit = f"hit@{k}"
    print(f"\n{'variant':13}{'window':>12}{'chunks':>9}{'dev':>8}{'eval':>8}{'eval hi':>9}{'eval gu':>9}")
    print("-" * 68)
    for label, row in result["variants"].items():
        window = ",".join(f"{lg}={n}" for lg, n in row["child_sentences"].items()) or "-"
        lang = row["eval"]["by_language"]
        print(
            f"{label:13}{window:>12}{row['chunks_indexed']:9d}"
            f"{row['dev']['overall'][hit]:8.4f}{row['eval']['overall'][hit]:8.4f}"
            f"{lang.get('hi', {}).get(hit, 0):9.4f}{lang.get('gu', {}).get(hit, 0):9.4f}"
        )
    print(f"\nwindows chosen on dev: {result['dev_selected_child_sentences']}")
    print(f"\n{'comparison (held-out eval)':32}{'overall':>21}{'hi':>21}{'gu':>21}")
    print("-" * 95)
    for name, delta in result["comparisons"].items():
        cells = [delta["overall"]] + [delta["by_language"][lg] for lg in ("hi", "gu")]
        print(f"{name:32}" + "".join(f"{c['mean_diff']:+13.4f} p={c['p_value']:<7.4f}" for c in cells))
    if args.output:
        print(f"\nWrote {args.output}")


def _cmd_fusion_sweep(args: argparse.Namespace) -> None:
    from eval.sweep_fusion import sweep_fusion_weight

    if not args.dev_file.exists():
        raise SystemExit(f"Dev file not found: {args.dev_file}")

    result = sweep_fusion_weight(
        args.dev_file,
        top_k=args.top_k,
        output_path=args.output,
    )
    print(f"{'w_dense':>8}{'hi hit@5':>10}{'gu hit@5':>10}{'overall':>10}")
    print("-" * 38)
    for row in result["sweep"]:
        print(
            f"{row['dense_weight']:8.2f}{row[f'hi_hit@{args.top_k}']:10.4f}"
            f"{row[f'gu_hit@{args.top_k}']:10.4f}{row[f'overall_hit@{args.top_k}']:10.4f}"
        )
    best = result["best_dense_weight"]
    print("\nbest on dev: " + ", ".join(f"{lg}={w:.2f}" for lg, w in best.items()))
    if args.output:
        print(f"Wrote {args.output}")


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
        reuse_index=args.reuse_index,
        ablate_dedupe=args.ablate_dedupe,
        output_path=args.output,
    )
    print(f"{'strategy':14} {'chunks':>8} {'hit@5':>8} {'mrr':>8} {'hi hit@5':>9} {'gu hit@5':>9}")
    print("-" * 62)
    for name in args.strategies:
        row = results.get(name)
        if not row:
            continue
        overall, by_lang = row["overall"], row["by_language"]
        print(
            f"{name:14} {row.get('chunks_indexed') or 0:8d} "
            f"{overall[f'hit@{args.top_k}']:8.4f} {overall['mrr']:8.4f} "
            f"{by_lang.get('hi', {}).get(f'hit@{args.top_k}', 0):9.4f} "
            f"{by_lang.get('gu', {}).get(f'hit@{args.top_k}', 0):9.4f}"
        )
    print(f"\nWrote {args.output}")


def _cmd_retriever_compare(args: argparse.Namespace) -> None:
    from eval.compare_retrievers import compare_retrievers

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")

    results = compare_retrievers(
        tuple(args.retrievers),
        args.eval_file,
        top_k=args.top_k,
        baseline=args.baseline,
        bootstrap_resamples=args.bootstrap_resamples,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


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
        "--mode",
        choices=["fast", "quality"],
        default="fast",
        help="fast: local extractive, <200ms. quality: LLM tool-calling harness.",
    )
    p_query.add_argument(
        "--local-only",
        action="store_true",
        help="Never call a remote model (retrieval + extractive answer only)",
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
        help="Build the held-out and dev query sets + split.json from MSMARCO-XI",
    )
    p_eval_build.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/queries.jsonl"),
    )
    p_eval_build.add_argument(
        "--dev-output",
        type=Path,
        default=Path("data/eval/dev_queries.jsonl"),
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
        default=["fixed", "semantic", "metadata", "recursive", "parent_child", "token_window"],
        choices=["fixed", "semantic", "metadata", "recursive", "parent_child", "token_window"],
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
        choices=["fixed", "semantic", "metadata", "recursive", "parent_child", "token_window"],
    )
    p_chunk_compare.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
    )
    p_chunk_compare.add_argument(
        "--reuse-index",
        action="store_true",
        help="Score against indexes already built for each strategy, skipping ingest",
    )
    p_chunk_compare.add_argument(
        "--ablate-dedupe",
        action="store_true",
        help="Also score each strategy without passage-level candidate collapsing",
    )
    p_chunk_compare.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/chunk-compare.json"),
    )
    p_chunk_compare.set_defaults(func=_cmd_chunk_compare)

    p_child_sweep = sub.add_parser(
        "child-sweep",
        help="Sweep the parent_child child window size, reported per language",
    )
    p_child_sweep.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4])
    p_child_sweep.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/eval/queries.jsonl"),
    )
    p_child_sweep.add_argument(
        "--dev-file",
        type=Path,
        default=Path("data/eval/dev_queries.jsonl"),
        help="Slice the per-language window size is chosen on (never eval)",
    )
    p_child_sweep.add_argument("--top-k", type=int, default=5)
    p_child_sweep.add_argument(
        "--reuse-index",
        action="store_true",
        help="Score against indexes already built for each size, skipping ingest",
    )
    p_child_sweep.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/child-size-sweep.json"),
    )
    p_child_sweep.set_defaults(func=_cmd_child_sweep)

    p_fusion_sweep = sub.add_parser(
        "fusion-sweep",
        help="Sweep the hybrid dense/lexical weight on the dev slice",
    )
    p_fusion_sweep.add_argument(
        "--dev-file",
        type=Path,
        default=Path("data/eval/dev_queries.jsonl"),
    )
    p_fusion_sweep.add_argument("--top-k", type=int, default=5)
    p_fusion_sweep.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/fusion-sweep.json"),
    )
    p_fusion_sweep.set_defaults(func=_cmd_fusion_sweep)

    p_retriever_compare = sub.add_parser(
        "retriever-compare",
        help="Compare dense / sparse / hybrid retrieval on the held-out eval set",
    )
    p_retriever_compare.add_argument(
        "--retrievers",
        nargs="+",
        default=["dense", "sparse", "hybrid"],
        choices=["dense", "sparse", "hybrid"],
    )
    p_retriever_compare.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/eval/queries.jsonl"),
    )
    p_retriever_compare.add_argument("--top-k", type=int, default=5)
    p_retriever_compare.add_argument(
        "--baseline",
        default="dense",
        choices=["dense", "sparse", "hybrid"],
    )
    p_retriever_compare.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p_retriever_compare.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/retriever-compare.json"),
    )
    p_retriever_compare.set_defaults(func=_cmd_retriever_compare)

    p_guardrail_calibrate = sub.add_parser(
        "guardrail-calibrate",
        help="Calibrate GUARDRAIL_MIN_SCORE from answerable vs abstain queries",
    )
    p_guardrail_calibrate.add_argument(
        "--answerable-file",
        type=Path,
        default=Path("data/eval/queries.jsonl"),
    )
    p_guardrail_calibrate.add_argument(
        "--abstain-file",
        type=Path,
        default=Path("data/eval/abstain_queries.jsonl"),
    )
    p_guardrail_calibrate.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/guardrail-calibration.json"),
    )
    p_guardrail_calibrate.add_argument("--top-k", type=int, default=5)
    p_guardrail_calibrate.set_defaults(func=_cmd_guardrail_calibrate)

    p_bench = sub.add_parser("bench", help="Latency benchmark (P50/P70/P100)")
    p_bench.add_argument(
        "--queries",
        type=int,
        default=300,
        help="Answerable queries to sample from the held-out set (abstain set is always added)",
    )
    p_bench.add_argument(
        "--modes",
        nargs="+",
        default=["fast"],
        choices=["fast", "quality"],
        help="Which pipeline paths to measure",
    )
    p_bench.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory of .wav clips for the voice end-to-end track",
    )
    p_bench.add_argument(
        "--output",
        type=Path,
        default=Path("data/bench/latency.json"),
    )
    p_bench.set_defaults(func=_cmd_bench)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe audio to text (hi/gu)")
    p_transcribe.add_argument("audio", type=Path)
    p_transcribe.add_argument("--language", choices=["hi", "gu"], default="hi")
    p_transcribe.set_defaults(func=_cmd_transcribe)

    p_voice_query = sub.add_parser("voice-query", help="Transcribe audio then run RAG")
    p_voice_query.add_argument("audio", type=Path)
    p_voice_query.add_argument("--language", choices=["hi", "gu"], default="hi")
    p_voice_query.add_argument("--top-k", type=int, default=None)
    p_voice_query.add_argument(
        "--mode",
        choices=["fast", "quality"],
        default="fast",
    )
    p_voice_query.add_argument(
        "--local-only",
        action="store_true",
        help="Never call a remote model (retrieval + extractive answer only)",
    )
    p_voice_query.set_defaults(func=_cmd_voice_query)

    p_serve = sub.add_parser("serve", help="Start HTTP API")
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
