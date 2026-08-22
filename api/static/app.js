/* Voice RAG demo UI.
 *
 * Pure client of the HTTP API. No retrieval or model logic lives here.
 *
 * The answer arrives in two tiers, which is what the pipeline actually does:
 *   tier 1  retrieval + guardrails + extractive answer, fully local
 *   tier 2  the same answer rewritten by an LLM that may search again
 * Tier 1 is what the 200ms budget is measured against. Tier 2 is optional and
 * replaces the text only if it succeeds, so a failure is invisible to the user.
 */

const LATENCY_BUDGET_MS = 200;

const $ = (sel) => document.querySelector(sel);

const el = {
  config: $("#config"),
  mic: $("#mic"),
  micLabel: $("#mic-label"),
  question: $("#question"),
  ask: $("#ask"),
  result: $("#result"),
  transcript: $("#transcript"),
  answer: $("#answer"),
  answerEn: $("#answer-en"),
  refusal: $("#refusal"),
  stages: $("#stages"),
  stagesHead: $("#stages-head"),
  sources: $("#sources"),
  sourcesHead: $("#sources-head"),
  badgePath: $("#badge-path"),
  badgeLang: $("#badge-lang"),
  detected: $("#detected"),
  tier1: $("#tier1"),
  tier1Ms: $("#tier1-ms"),
  tier2: $("#tier2"),
  tier2Ms: $("#tier2-ms"),
  budgetFill: $("#budget-fill"),
  budgetLegend: $("#budget-legend"),
  mHit: $("#m-hit"),
  mHitNote: $("#m-hit-note"),
  mLatency: $("#m-latency"),
  mLatencyNote: $("#m-latency-note"),
  mGuard: $("#m-guard"),
  mGuardNote: $("#m-guard-note"),
  mChunks: $("#m-chunks"),
  provenance: $("#provenance"),
  session: $("#session"),
  sCount: $("#s-count"),
  sP50: $("#s-p50"),
  sP70: $("#s-p70"),
  sP100: $("#s-p100"),
  sAbstain: $("#s-abstain"),
  sNote: $("#s-note"),
};

const state = { busy: false };

/* Display names. The trace keys stay as they are, because every published
 * number in docs/latency.md is keyed on them; only the labels change, and they
 * use the vocabulary of the task brief rather than internal stage names. */
const STAGE_LABELS = {
  input_guard: "Safety check",
  retrieve: "Retrieval · vector DB",
  grounding_guard: "Grounding check",
  answer_fast: "Answer generation",
  answer_fast_fallback: "Answer generation · fallback",
  answer_quality_cached: "LLM rewrite · cached",
  faithfulness: "Hallucination check",
  verify_citations: "Citation check",
};

// answer_quality carries the provider, e.g. "answer_quality[openai]".
const stageLabel = (name) =>
  STAGE_LABELS[name] ||
  (name.startsWith("answer_quality")
    ? `LLM rewrite${name.slice(14).replace(/[[\]]/g, " ").trimEnd()}`
    : name.replace(/_/g, " "));

/* Why an answer was declined, in plain words. */
const REASON_LABELS = {
  unsafe_content: "unsafe request, refused before searching",
  blocked_intent: "prompt injection, refused before searching",
  unsupported_language: "not Hindi or Gujarati",
  empty_query: "empty question",
  short_query: "question too short",
  no_context: "nothing retrieved",
  low_confidence: "not enough supporting evidence in the corpus",
  ungrounded_answer: "answer was not supported by the retrieved passages",
};

// Devanagari and Gujarati are disjoint Unicode blocks, so the script the user
// typed in identifies the language. The server does the same detection; this
// copy only drives the live hint under the box.
const SCRIPTS = [["hi", "हिन्दी", 0x0900, 0x097f], ["gu", "ગુજરાતી", 0x0a80, 0x0aff]];
const LANGUAGE_NAMES = { hi: "हिन्दी", gu: "ગુજરાતી" };

function detectLanguage(text) {
  const counts = { hi: 0, gu: 0 };
  for (const char of text) {
    const code = char.codePointAt(0);
    for (const [code_, , low, high] of SCRIPTS) {
      if (code >= low && code <= high) { counts[code_] += 1; break; }
    }
  }
  if (!counts.hi && !counts.gu) return null;
  return counts.gu > counts.hi ? "gu" : "hi";
}

const fmtMs = (v) => `${Number(v).toFixed(1)}ms`;
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function setBusy(busy) {
  state.busy = busy;
  el.ask.disabled = busy;
  el.mic.disabled = busy;
  el.ask.textContent = busy ? "Working..." : "Ask";
}

function setTier(node, msNode, tierState, ms) {
  node.dataset.state = tierState;
  msNode.textContent = ms === null ? "-" : fmtMs(ms);
}

// Draws the pipeline time against the 200ms target.
function renderBudget(pipelineMs) {
  const pct = Math.min((pipelineMs / LATENCY_BUDGET_MS) * 100, 100);
  const over = pipelineMs > LATENCY_BUDGET_MS;
  el.budgetFill.style.width = `${Math.max(pct, 1)}%`;
  el.budgetFill.classList.toggle("is-over", over);
  el.budgetLegend.innerHTML = over
    ? `<b>${fmtMs(pipelineMs)}</b> of ${LATENCY_BUDGET_MS}ms budget · <span class="over">over</span>`
    : `<b>${fmtMs(pipelineMs)}</b> of ${LATENCY_BUDGET_MS}ms budget · ${(100 - pct).toFixed(0)}% unused`;
}

function renderStages(timings, total) {
  const entries = Object.entries(timings || {});
  el.stagesHead.hidden = !entries.length;
  if (!entries.length) {
    el.stages.innerHTML = "";
    return;
  }
  const max = Math.max(...entries.map(([, v]) => v), 0.01);
  el.stages.innerHTML = entries
    .map(([name, ms]) => {
      const width = Math.max((ms / max) * 100, 1.5);
      const slow = ms > total * 0.5;
      return `<div class="stage">
        <span class="stage-name">${esc(stageLabel(name))}</span>
        <span class="stage-bar"><span class="stage-fill${slow ? " is-slow" : ""}" style="width:${width}%"></span></span>
        <span class="stage-ms">${fmtMs(ms)}</span>
      </div>`;
    })
    .join("");
}

function renderSources(sources, citations) {
  const cited = new Set(citations || []);
  // A query refused before retrieval has no passages; the heading would
  // otherwise sit above nothing and read as a rendering failure.
  el.sourcesHead.hidden = !(sources || []).length;
  el.sources.innerHTML = (sources || [])
    .map((s, i) => {
      const c = s.components || {};
      const parts = [`#${i + 1}`, esc(s.document_id)];
      if (c.dense_score !== undefined) parts.push(`dense ${c.dense_score.toFixed(3)}`);
      if (c.sparse_score !== undefined) parts.push(`bm25 ${c.sparse_score.toFixed(2)}`);
      if (cited.has(s.id)) parts.push(`<span class="cited">cited</span>`);
      // The English is the passage's source text from MS MARCO-XI, not a
      // translation of what is shown above it.
      const en = s.text_en
        ? `<div class="source-en">${esc(s.text_en.slice(0, 260))}${s.text_en.length > 260 ? "..." : ""}</div>`
        : "";
      return `<div class="source">
        <div class="source-head">${parts.join(" · ")}</div>
        <div>${esc(s.text.slice(0, 260))}${s.text.length > 260 ? "..." : ""}</div>
        ${en}
      </div>`;
    })
    .join("");
}

// Tier 1: the local answer. This is the number the budget is measured against.
// Percentiles over this session's fast-path queries, computed the same way the
// benchmark does: nearest-rank on the sorted sample, P100 as the true maximum.
const session = { latencies: [], abstains: 0 };

function percentile(sorted, q) {
  if (!sorted.length) return null;
  return sorted[Math.min(Math.floor(sorted.length * q), sorted.length - 1)];
}

function recordQuery(data) {
  // total_ms is the server's own pipeline timing, so the figure excludes
  // network and matches what the published percentiles measure.
  if (typeof data.total_ms !== "number") return;
  session.latencies.push(data.total_ms);
  if (data.abstained) session.abstains += 1;

  const sorted = [...session.latencies].sort((a, b) => a - b);
  const n = sorted.length;
  el.session.hidden = false;
  el.sCount.textContent = `· ${n} ${n === 1 ? "query" : "queries"}`;
  el.sP50.textContent = percentile(sorted, 0.5).toFixed(1);
  el.sP70.textContent = percentile(sorted, 0.7).toFixed(1);
  el.sP100.textContent = sorted[n - 1].toFixed(1);
  el.sAbstain.textContent = String(session.abstains);
  el.sNote.textContent =
    `Live, this browser only. The figures above come from committed benchmark runs; ${n} ${n === 1 ? "query is" : "queries are"} far too few to compare against them.`;
}

function renderFast(data, transcript) {
  el.result.hidden = false;
  recordQuery(data);

  el.transcript.hidden = !transcript;
  if (transcript) el.transcript.innerHTML = `heard <b>${esc(transcript)}</b>`;

  const abstained = data.abstained;
  el.answer.hidden = abstained;
  el.refusal.hidden = !abstained;

  if (abstained) {
    const why = REASON_LABELS[data.guardrail_reason] || data.guardrail_reason || "";
    el.refusal.innerHTML = `<b>Declined · ${esc(why)}</b>${esc(data.answer || "")}`;
  } else {
    el.answer.textContent = data.answer || "";
  }

  // English source of the passage the answer was quoted from.
  const cited = (data.sources || [])[0];
  const showEn = !abstained && cited && cited.text_en;
  el.answerEn.hidden = !showEn;
  if (showEn) {
    el.answerEn.innerHTML =
      `<b>English source of the cited passage</b>${esc(cited.text_en)}`;
  }

  el.badgePath.textContent = abstained ? "Declined" : "Grounded";
  el.badgePath.className = abstained ? "tag tag-pink" : "tag";
  el.badgeLang.textContent = LANGUAGE_NAMES[data.language] || data.language || "";

  setTier(el.tier1, el.tier1Ms, "done", data.total_ms);
  setTier(el.tier2, el.tier2Ms, abstained ? "skipped" : "pending", null);

  renderBudget(data.total_ms);
  renderStages(data.timings_ms, data.total_ms);
  renderSources(data.sources, data.citations);
  el.result.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Tier 2: LLM rewrite. Silent on failure, because tier 1 already answered.
function renderGenerated(data, elapsedMs) {
  if (data.abstained || !data.answer) {
    setTier(el.tier2, el.tier2Ms, "skipped", null);
    return;
  }
  el.answer.textContent = data.answer;
  const top = (data.sources || [])[0];
  el.answerEn.hidden = !(top && top.text_en);
  if (top && top.text_en) {
    el.answerEn.innerHTML = `<b>English source of the cited passage</b>${esc(top.text_en)}`;
  }
  el.badgePath.textContent = "Generated";
  setTier(el.tier2, el.tier2Ms, data.path === "quality" ? "done" : "skipped", elapsedMs);
  renderSources(data.sources, data.citations);
}

function showError(message) {
  el.result.hidden = false;
  el.answer.hidden = true;
  el.refusal.hidden = false;
  el.refusal.innerHTML = `<b>Error</b>${esc(message)}`;
  setTier(el.tier1, el.tier1Ms, "idle", null);
  setTier(el.tier2, el.tier2Ms, "idle", null);
}

async function postQuery(question, mode) {
  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // No language field: the server reads it off the script.
    body: JSON.stringify({ question, mode }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// Tier 2 runs after tier 1 has already rendered, so the wait costs nothing.
async function polish(question) {
  const started = performance.now();
  try {
    const data = await postQuery(question, "quality");
    renderGenerated(data, performance.now() - started);
  } catch {
    setTier(el.tier2, el.tier2Ms, "skipped", null);
  }
}

async function askText() {
  const question = el.question.value.trim();
  if (!question || state.busy) return;
  setBusy(true);
  try {
    const fast = await postQuery(question, "fast");
    renderFast(fast, null);
    setBusy(false);
    if (!fast.abstained) await polish(question);
  } catch (err) {
    showError(err.message);
    setBusy(false);
  }
}

/* Recording ------------------------------------------------------------- */

let recorder = null;
let chunks = [];

// Container varies by browser: Chrome gives webm/opus, Safari mp4. The filename
// extension has to match or the STT provider misreads the bytes.
const EXT_BY_TYPE = [
  ["audio/webm", "webm"],
  ["audio/ogg", "ogg"],
  ["audio/mp4", "mp4"],
  ["audio/mpeg", "mp3"],
  ["audio/wav", "wav"],
];

const extensionFor = (mime) =>
  (EXT_BY_TYPE.find(([p]) => (mime || "").startsWith(p)) || [, "webm"])[1];

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    el.micLabel.textContent = "Microphone blocked";
    return;
  }

  chunks = [];
  // Hold the instance locally. stop() fires onstop asynchronously and the
  // module-level recorder is cleared before then, so reading it in the handler
  // would hit null and strand the upload.
  const active = new MediaRecorder(stream);
  recorder = active;

  active.ondataavailable = (e) => e.data.size && chunks.push(e.data);

  active.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: active.mimeType });
    if (!blob.size) {
      el.micLabel.textContent = "Tap to record";
      showError("No audio was captured. Check the microphone and try again.");
      return;
    }
    sendAudio(blob, extensionFor(active.mimeType));
  };

  active.onerror = (e) => {
    stream.getTracks().forEach((t) => t.stop());
    recorder = null;
    el.mic.classList.remove("is-recording");
    el.micLabel.textContent = "Tap to record";
    showError(`Recording failed: ${e.error?.name || "unknown error"}`);
  };

  active.start();
  el.mic.classList.add("is-recording");
  el.micLabel.textContent = "Tap to stop";
}

function stopRecording() {
  recorder?.stop();
  recorder = null;
  el.mic.classList.remove("is-recording");
  el.micLabel.textContent = "Transcribing...";
}

async function sendAudio(blob, extension) {
  setBusy(true);
  const form = new FormData();
  // No language field: Scribe detects it, and the transcript's script confirms.
  form.append("mode", "fast");
  form.append("audio", blob, `question.${extension}`);

  try {
    const res = await fetch("/voice-query", { method: "POST", body: form });
    const body = await res.text();
    let data;
    try {
      data = JSON.parse(body);
    } catch {
      throw new Error(`${res.status} ${body.slice(0, 200)}`);
    }
    if (!res.ok) throw new Error(data.detail || res.statusText);

    el.question.value = data.transcription;
    // STT is a network call and sits outside the 200ms budget by design.
    renderFast(data, `${data.transcription}  ·  STT ${fmtMs(data.stt_ms)} (outside budget)`);
    setBusy(false);
    if (!data.abstained) await polish(data.transcription);
  } catch (err) {
    showError(err.message);
    setBusy(false);
  } finally {
    el.micLabel.textContent = "Tap to record";
  }
}

el.mic.addEventListener("click", () => {
  if (state.busy) return;
  recorder ? stopRecording() : startRecording();
});

/* Wiring ----------------------------------------------------------------- */

el.ask.addEventListener("click", askText);
el.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) askText();
});

// Live hint, so it is obvious the language was read rather than assumed.
el.question.addEventListener("input", () => {
  const lang = detectLanguage(el.question.value);
  el.detected.innerHTML = lang
    ? `Detected <b>${LANGUAGE_NAMES[lang]}</b>`
    : "Hindi and Gujarati are detected automatically.";
});

// The page states the configuration it is actually serving, rather than
// repeating whatever the README claimed at the time it was written.
async function loadConfig() {
  try {
    const h = await (await fetch("/health")).json();
    el.mChunks.textContent = h.indexed_chunks.toLocaleString("en-US");
    el.config.textContent = [
      `${h.chunking} chunking`,
      `${h.retriever} retrieval`,
      `${h.fusion.toUpperCase()} fusion`,
      h.embedding_preset,
      `${h.stt_provider} speech-to-text`,
    ].join("  ·  ");
  } catch {
    el.config.textContent = "API unreachable. Start the server with ./hhgoa serve";
  }
}

// Measured values read from run artifacts, never constants. A missing artifact
// leaves the dash in place rather than showing an invented number.
async function loadStats() {
  let s;
  try {
    s = await (await fetch("/stats")).json();
  } catch {
    return;
  }

  const { retrieval: r, latency: l, guardrail: g } = s;

  if (r.hit_at_5 !== null) {
    el.mHit.textContent = r.hit_at_5.toFixed(3);
    const gain = r.baseline_hit_at_5 !== null
      ? `, +${((r.hit_at_5 - r.baseline_hit_at_5) * 100).toFixed(1)} points over dense-only`
      : "";
    el.mHitNote.textContent = `hit@5 on ${r.queries} held-out queries${gain}`;
  }

  if (l.p50_ms !== null) {
    el.mLatency.innerHTML = [["P50", l.p50_ms], ["P70", l.p70_ms], ["P100", l.p100_ms]]
      .map(([name, ms]) => `<span><b>${name}</b>${ms}<small> ms</small></span>`)
      .join("");
    el.mLatencyNote.textContent =
      `${l.queries} queries, worst case inside a ${l.budget_ms} ms budget`;
  }

  if (g.false_abstain_rate !== null) {
    el.mGuard.innerHTML = `${(g.false_abstain_rate * 100).toFixed(1)}<small>%</small>`;
    // Pair the two rates: a low false-abstain rate alone says nothing about
    // how many unanswerable questions the gate actually caught.
    el.mGuardNote.textContent = g.abstain_recall !== null
      ? `answerable questions wrongly refused, while catching ${(g.abstain_recall * 100).toFixed(0)}% of unanswerable ones`
      : "answerable questions wrongly refused";
  }

  if (s.sources.length) {
    el.provenance.textContent = `Measured, not claimed. Source files: ${s.sources.join("  ·  ")}`;
  }
}

loadConfig();
loadStats();
