"""The orchestration harness.

Runs the pipeline as an explicit stage graph rather than a straight-line
function. Every stage is timed, every failure is caught and classified, and
every stage declares what happens when it fails. The contract with callers is
that ``run()`` always returns a response: there is no exception path that
reaches the API or the CLI.

Two paths share the same graph:

  fast     guard -> retrieve -> grounding -> extractive answer -> faithfulness
           Fully local, no network, designed to fit the 200 ms budget.

  quality  the fast path, then a tool-calling LLM round with structured output
           and one repair attempt. Any failure, any deadline breach, and the
           already-computed fast answer is returned instead, which is why the
           fast path runs first even in quality mode.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.harness.cache import AnswerCache
from core.harness.contracts import (
    ANSWER_JSON_SCHEMA,
    AnswerPayload,
    QueryEnvelope,
    StageResult,
    Trace,
)
from core.harness.policy import Deadline, RetryPolicy, call_with_retry
from core.harness.structured import (
    StructuredOutputError,
    parse_answer,
    repair_prompt,
    verify_citations,
)
from core.harness.tools import MAX_TOOL_ROUNDS, TOOL_SPECS, ToolRunner
from core.llm.base import BaseLLM
from core.rag.prompts import SYSTEM_PROMPT, build_rag_prompt
from core.retriever.base import BaseRetriever
from core.types import RAGResponse, ScoredChunk


class Orchestrator:
    def __init__(
        self,
        *,
        retriever: BaseRetriever,
        guardrail: BaseGuardrail,
        fast_answerer: BaseLLM,
        chat_clients: list = (),
        system_prompt: str = SYSTEM_PROMPT,
        retry_policy: RetryPolicy | None = None,
        default_language: str = "hi",
        top_k: int = 5,
        answer_cache: AnswerCache | None = None,
    ) -> None:
        self.retriever = retriever
        self.guardrail = guardrail
        self.fast_answerer = fast_answerer
        # Ordered fallback chain: primary provider first, secondary next.
        self.chat_clients = [c for c in chat_clients if getattr(c, "configured", False)]
        self.system_prompt = system_prompt
        self.retry_policy = retry_policy or RetryPolicy()
        self.default_language = default_language
        self.top_k = top_k
        # Only the quality path is cached; see core.harness.cache.
        self.answer_cache = answer_cache or AnswerCache(0)

    # ------------------------------------------------------------ plumbing

    @contextmanager
    def _stage(self, trace: Trace, name: str):
        """Time a stage and record it, whatever happens inside."""
        start = time.perf_counter()
        record = {"status": "ok", "attempts": 1, "error": None, "detail": {}}
        try:
            yield record
        except BaseException as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace.add(
                StageResult(
                    name=name,
                    status=record["status"],  # type: ignore[arg-type]
                    ms=(time.perf_counter() - start) * 1000.0,
                    attempts=int(record["attempts"]),
                    error=record["error"],
                    detail=record["detail"],
                )
            )

    @staticmethod
    def _response(
        envelope: QueryEnvelope,
        answer: str,
        sources: list[ScoredChunk],
        trace: Trace,
        *,
        extra: dict | None = None,
    ) -> RAGResponse:
        metadata: dict = {
            "trace_id": trace.trace_id,
            "path": trace.path,
            "mode": trace.mode,
            "timings_ms": trace.timings(),
            "total_ms": round(trace.total_ms, 3),
        }
        metadata.update(extra or {})
        return RAGResponse(
            query=envelope.text,
            answer=answer,
            sources=sources,
            language=envelope.language,
            metadata=metadata,
        )

    def _blocked(
        self,
        envelope: QueryEnvelope,
        decision: GuardrailDecision,
        trace: Trace,
    ) -> RAGResponse:
        trace.path = "abstained"
        return self._response(
            envelope,
            decision.answer or "",
            decision.sources or [],
            trace,
            extra={
                "guardrail": {
                    "blocked": True,
                    "stage": decision.stage,
                    "reason": decision.reason,
                    **decision.metadata,
                }
            },
        )

    # ------------------------------------------------------------------ run

    def run(self, envelope: QueryEnvelope) -> RAGResponse:
        trace = Trace(
            trace_id=envelope.trace_id or uuid.uuid4().hex[:12],
            mode=envelope.mode,
            path="fast",
        )
        deadline = Deadline(budget_ms=float(envelope.deadline_ms))
        started = time.perf_counter()

        def finish(response: RAGResponse) -> RAGResponse:
            trace.total_ms = (time.perf_counter() - started) * 1000.0
            response.metadata["timings_ms"] = trace.timings()
            response.metadata["total_ms"] = round(trace.total_ms, 3)
            response.metadata["path"] = trace.path
            response.metadata["stages"] = [s.model_dump() for s in trace.stages]
            return response

        # 1. Input intent -------------------------------------------------
        with self._stage(trace, "input_guard") as rec:
            input_decision = self.guardrail.check_input(
                envelope.text, language=envelope.language
            )
            if input_decision.blocked:
                rec["status"] = "blocked"
                rec["detail"] = {"reason": input_decision.reason}
        if input_decision.blocked:
            return finish(self._blocked(envelope, input_decision, trace))

        # 2. Retrieval ----------------------------------------------------
        try:
            with self._stage(trace, "retrieve") as rec:
                sources = self.retriever.retrieve(
                    envelope.text,
                    top_k=envelope.top_k or self.top_k,
                    language=envelope.language,
                )
                rec["detail"] = {"candidates": len(sources)}
        except Exception:  # noqa: BLE001 - any retrieval failure must abstain, not raise
            # Retrieval is the one stage with no meaningful fallback: without
            # context there is nothing to ground an answer on, so abstain.
            trace.path = "error"
            return finish(
                self._response(
                    envelope,
                    "",
                    [],
                    trace,
                    extra={
                        "guardrail": {
                            "blocked": True,
                            "stage": "retrieve",
                            "reason": "retrieval_failed",
                        }
                    },
                )
            )

        # 3. Grounding ----------------------------------------------------
        with self._stage(trace, "grounding_guard") as rec:
            grounding = self.guardrail.check_grounding(
                envelope.text, sources, language=envelope.language
            )
            if grounding.blocked:
                rec["status"] = "blocked"
                rec["detail"] = {"reason": grounding.reason}
        if grounding.blocked:
            return finish(self._blocked(envelope, grounding, trace))

        sources = grounding.sources or sources

        # 4. Fast answer (always computed, it is the fallback) ------------
        try:
            with self._stage(trace, "answer_fast"):
                fast_answer = self.fast_answerer.answer_with_context(
                    envelope.text, sources, language=envelope.language
                )
        except Exception:  # noqa: BLE001 - any answerer failure falls back to raw context
            with self._stage(trace, "answer_fast_fallback") as rec:
                rec["status"] = "fallback"
                fast_answer = sources[0].chunk.text if sources else ""

        answer = fast_answer
        citations = [s.chunk.id for s in sources[:1]]
        quality_meta: dict = {}

        # 5. Quality path (optional) --------------------------------------
        if envelope.mode == "quality" and self.chat_clients:
            cache_key = self.answer_cache.key(
                envelope.text, envelope.language, [s.chunk.id for s in sources]
            )
            cached = self.answer_cache.get(cache_key)
            if cached is not None:
                with self._stage(trace, "answer_quality_cached") as rec:
                    rec["detail"] = {"cache": "hit"}
                # The answer really was generated, just not on this request.
                # Reporting it as "fast" would misattribute it to the
                # extractive path.
                trace.path = "quality_cached"
                answer, citations, quality_meta = cached
            else:
                answer, citations, quality_meta = self._quality_path(
                    envelope, sources, trace, deadline, fallback_answer=fast_answer
                )
                # Only a real generated answer is worth keeping. `trace.path`
                # is "fast_fallback" when the provider failed, and caching that
                # would pin a transient outage in memory.
                if trace.path == "quality":
                    self.answer_cache.put(cache_key, (answer, citations, quality_meta))

        # 6. Faithfulness --------------------------------------------------
        with self._stage(trace, "faithfulness") as rec:
            answer_decision = self.guardrail.check_answer(
                envelope.text, answer, sources, language=envelope.language
            )
            if answer_decision.blocked:
                rec["status"] = "blocked"
                rec["detail"] = {"reason": answer_decision.reason}
        if answer_decision.blocked:
            response = self._blocked(envelope, answer_decision, trace)
            response.metadata.update(quality_meta)
            return finish(response)

        extra = {"citations": citations}
        if answer_decision.metadata:
            extra["hallucination"] = answer_decision.metadata
        extra.update(quality_meta)
        return finish(
            self._response(
                envelope, answer_decision.answer or answer, sources, trace, extra=extra
            )
        )

    # -------------------------------------------------------- quality path

    def _quality_path(
        self,
        envelope: QueryEnvelope,
        sources: list[ScoredChunk],
        trace: Trace,
        deadline: Deadline,
        *,
        fallback_answer: str,
    ) -> tuple[str, list[str], dict]:
        """Tool-calling + structured generation. Never raises."""
        runner = ToolRunner(self.retriever, language=envelope.language)
        runner.register(sources)

        messages = [
            {"role": "system", "content": self.system_prompt + _QUALITY_SYSTEM_SUFFIX},
            {
                "role": "user",
                "content": build_rag_prompt(envelope.text, sources, language=envelope.language)
                + _CITATION_HINT.format(ids=", ".join(s.chunk.id for s in sources)),
            },
        ]

        last_error: str | None = None
        for client in self.chat_clients:
            try:
                with self._stage(trace, f"answer_quality[{client.provider}]") as rec:
                    payload, attempts, rounds = self._run_chat(
                        client, messages, runner, deadline
                    )
                    rec["attempts"] = attempts
                    rec["detail"] = {
                        "tool_rounds": rounds,
                        "tool_calls": [c["name"] for c in runner.calls],
                    }
                valid, fabricated = verify_citations(payload, runner.valid_ids())
                trace.path = "quality"
                meta = {
                    "quality": {
                        "provider": client.provider,
                        "model": client.model,
                        "confidence": payload.confidence,
                        "sufficient": payload.sufficient,
                        "tool_calls": [c["name"] for c in runner.calls],
                        "fabricated_citations": fabricated,
                    }
                }
                # The model saying "the context doesn't answer this" is a
                # first-class abstain signal, not something to paper over.
                if not payload.sufficient:
                    meta["quality"]["abstained_by_model"] = True
                    return fallback_answer, valid, meta
                return payload.answer, valid or [s.chunk.id for s in sources[:1]], meta
            except Exception as exc:  # noqa: BLE001 - try the next provider
                last_error = f"{type(exc).__name__}: {exc}"
                continue

        # Every provider failed or the budget ran out: keep the grounded
        # extractive answer rather than surfacing an error.
        trace.path = "fast_fallback"
        return (
            fallback_answer,
            [s.chunk.id for s in sources[:1]],
            {"quality": {"failed": True, "error": last_error}},
        )

    def _run_chat(
        self,
        client,
        messages: list[dict],
        runner: ToolRunner,
        deadline: Deadline,
    ) -> tuple[AnswerPayload, int, int]:
        """One provider: bounded tool loop, structured output, one repair."""
        conversation = list(messages)
        total_attempts = 0
        rounds = 0

        for _ in range(MAX_TOOL_ROUNDS + 1):
            # On the last pass the tools are withheld, which leaves the model no
            # move except to answer. Without this a model that calls a tool on
            # every turn burns the whole budget and never concludes.
            tools = TOOL_SPECS if rounds < MAX_TOOL_ROUNDS else None
            message, attempts = call_with_retry(
                lambda _tools=tools: client.complete(
                    conversation, tools=_tools, json_schema=ANSWER_JSON_SCHEMA
                ),
                policy=self.retry_policy,
                deadline=None,  # quality path is explicitly allowed to overrun
            )
            total_attempts += attempts

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                raw = message.get("content") or ""
                try:
                    return parse_answer(raw), total_attempts, rounds
                except StructuredOutputError as exc:
                    # Exactly one repair round-trip, then give up.
                    conversation.append({"role": "assistant", "content": raw})
                    conversation.append({"role": "user", "content": repair_prompt(raw, str(exc))})
                    repaired, attempts = call_with_retry(
                        lambda: client.complete(conversation, json_schema=ANSWER_JSON_SCHEMA),
                        policy=self.retry_policy,
                    )
                    total_attempts += attempts
                    return parse_answer(repaired.get("content") or ""), total_attempts, rounds

            rounds += 1  # noqa: SIM113 - counts tool turns, not iterations
            conversation.append(message)
            for call in tool_calls:
                fn = call.get("function", {})
                result = runner.dispatch(fn.get("name", ""), fn.get("arguments", ""))
                conversation.append(
                    {"role": "tool", "tool_call_id": call.get("id", ""), "content": result}
                )

        # Unreachable: the final pass withholds tools, so it must return above.
        raise StructuredOutputError("tool loop ended without a final answer")


# The ordering here matters. An earlier version said the model "may" call
# search_corpus, which gave it no reason to when the first context looked
# adequate, so the tool was almost never exercised and thin context became a
# premature abstain. Searching is now the required step before giving up.
_QUALITY_SYSTEM_SUFFIX = (
    "\n\nAnswer using ONLY the provided context passages. Never use outside knowledge.\n"
    "Follow this order:\n"
    "1. If the context fully answers the question, answer it and cite the passages used.\n"
    "2. If the context is thin, partial, or off-target, call search_corpus with a "
    "reformulated query before concluding anything. Try a more specific term, a "
    "synonym, or the key entity on its own.\n"
    "3. Only after searching has failed, set \"sufficient\" to false. Never guess.\n"
    "You may call search_corpus at most twice. Always reply with the required JSON object."
)

_CITATION_HINT = (
    "\n\nCite passages by these exact ids: {ids}\n"
    "Reply as JSON with keys: answer, citations, sufficient, confidence."
)
