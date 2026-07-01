"""
Stage 30 - Generate grounded radiology QA (local LLM)
=====================================================
Expands a forget/retain pair pool into the V3-style QA benchmark, grounded on
real MIMIC report text.
API. Default mode is dry-run (sample plan + prompt preview, no model).

Mirrors MedicalGPT_V3 output for each coupling:
  outputs/<name>.json            master file (metadata + full/forget/retain/
                                 general lists with rich provenance)
  outputs/<name>/                split dir with 6 clean {question,answer} files:
     forget.json retain.json forget_rewrite.json retain_rewrite.json
     general.json full.json

Per patient anchor (forget & retain) -> qa_per_patient (default 10) QA, each
{question, answer, question_rewrite}. Identity = "Patient <subject_id>" + real
sex. Answers grounded ONLY in the anchor study's INDICATION/FINDINGS/IMPRESSION
(de-id placeholders sanitized to [redacted] before prompting); NO treatment /
prognosis / facts beyond the radiology report; never leak `___` or [redacted].
General QA = finding-level radiology knowledge (no patient).

HARD note: forget anchors are byte-identical to easy, but hard QA is generated
fresh (aligned with the hard retain) — easy forget QA is NOT reused.

Usage:
  python 30_generate_qa.py --coupling easy            # dry-run
  python 30_generate_qa.py --coupling easy --backend hf-batched \
      --model-path /path/to/local-model \
      --hf-auto-class image-text-to-text --dtype bfloat16 --device 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MIMIC_ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(MIMIC_ROOT, "config")
PROCESSED = os.path.join(MIMIC_ROOT, "processed")
POOLS = os.path.join(MIMIC_ROOT, "pools")
OUTPUTS = os.path.join(MIMIC_ROOT, "outputs")

DEFAULT_REPORTS = os.path.join(PROCESSED, "reports.json")
DEFAULT_TAXONOMY = os.path.join(CONFIG, "finding_taxonomy.yaml")
# INDICATION included so the "indication" intent is grounded; placeholders are
# sanitized before the text ever reaches the model.
GROUND_SECTIONS = ["INDICATION", "FINDINGS", "IMPRESSION"]


def sanitize(text):
    """Replace MIMIC de-id placeholders (___ / ___F) so they never reach the LLM."""
    return re.sub(r"_{2,}", "[redacted]", text)

# radiology QA intents (grounded, report-only). Kept finite for diversity.
INTENTS = [
    "the main positive finding on this chest X-ray",
    "whether the target finding is present and where (laterality/location)",
    "the severity or extent of the target finding as described",
    "any change of the target finding compared with prior studies (trend)",
    "other findings or devices mentioned in the report",
    "the overall impression of the study",
    "any explicitly stated follow-up or recommendation (only if present)",
    "what the report says is NOT present (pertinent negatives)",
    "the indication / reason the study was performed (if stated)",
    "a concise summary of this patient's chest X-ray findings",
]


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_finding_patterns(path):
    with open(path, encoding="utf-8") as fh:
        tax = yaml.safe_load(fh)
    out = {}
    for finding, cfg in (tax.get("findings") or {}).items():
        if finding == "Support Devices":
            continue
        pats = []
        for phrase in cfg.get("prescreen_patterns") or []:
            if phrase:
                pats.append(re.compile(re.escape(phrase), re.I))
        if pats:
            out[finding] = pats
    return out


def load_support_patterns(path):
    with open(path, encoding="utf-8") as fh:
        tax = yaml.safe_load(fh)
    cfg = ((tax.get("findings") or {}).get("Support Devices") or {})
    pats = []
    for phrase in cfg.get("prescreen_patterns") or []:
        if phrase:
            pats.append((phrase, re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", re.I)))
    return pats


def mentioned_findings(text, findings, finding_patterns):
    hits = []
    for finding in sorted(findings or []):
        if any(p.search(text or "") for p in finding_patterns.get(finding, [])):
            hits.append(finding)
    return hits


def mentioned_support_terms(text, support_patterns):
    return sorted({term for term, pat in support_patterns if pat.search(text or "")})


def report_sections(report, sections):
    raw = report.get("sections") or {}
    out = {}
    for s in sections:
        v = raw.get(s)
        if isinstance(v, str) and v.strip():
            out[s] = re.sub(r"\s+", " ", v).strip()
    return out


def grounding_text(report):
    secs = report_sections(report, GROUND_SECTIONS)
    return sanitize(" ".join(f"{k}: {v}" for k, v in secs.items()))


# ============================================================
# Prompts
# ============================================================

# pair-aligned intents (finding-centric). For a HARD pair, forget & retain share
# the same target_finding, so both get the IDENTICAL intent list -> they ask the
# same aspects of the same finding -> higher answer overlap, still grounded.
# finding-centric intents only -> answerable from ANY same-finding report (avoids
# section/organ-dependent intents like indication/cardiomediastinal/impression that
# many reports do not support, which previously left qid-slots permanently unfilled).
HARD_INTENT_TEMPLATES = [
    "presence and description of the {finding}",
    "severity or extent of the {finding}",
    "laterality of the {finding}",
    "anatomical location or distribution of the {finding}",
    "interval change of the {finding} versus any prior study",
    "other findings reported alongside the {finding}",
    "pertinent negative findings stated in the report",
    "the radiographic appearance or pattern of the {finding}",
    "the most significant abnormality described on this chest X-ray",
    "a concise summary of this patient's chest X-ray findings",
]


def aligned_intents_for(unit, n_qa):
    if n_qa > len(HARD_INTENT_TEMPLATES):
        raise ValueError(
            f"--align-intents needs qa_per_patient ({n_qa}) <= "
            f"{len(HARD_INTENT_TEMPLATES)} intent templates; add more templates first")
    f = unit["anchor"]["target_finding"]
    return [t.format(finding=f) for t in HARD_INTENT_TEMPLATES[:n_qa]]


def build_patient_prompt(unit, n_qa, aligned_intents=None):
    a = unit["anchor"]
    pid = a["patient_ref"]
    target_finding = a["target_finding"]
    disallowed = unit.get("disallowed_findings") or []
    disallowed_rule = ""
    if disallowed:
        disallowed_rule = (
            "\n10. EASY split leakage control: do NOT ask about or mention these "
            f"opposite-side findings in any question, rewrite, or answer: {', '.join(disallowed)}. "
            f"Every QA must stay focused on the target finding, {target_finding}."
        )
    sex_known = a["sex"] in ("male", "female")
    sex = a["sex"] if sex_known else "sex not stated"
    pronoun_rule = (
        f'Gendered pronouns are allowed because the report states the sex ({a["sex"]}).'
        if sex_known else
        f'The sex is NOT stated; NEVER use he/she/his/her/him — use only "{pid}" '
        'or "the patient".')
    if aligned_intents:
        intent_lines = "\n".join(f"   {i+1}. {it}" for i, it in enumerate(aligned_intents))
        intent_intro = (f"generate exactly {n_qa} radiology QA pairs for {pid} ({sex}), "
                        "one for EACH of the fixed clinical intents listed below, IN ORDER.")
        intent_rule = ("3. Address the intents in THIS fixed order (QA i = intent i); put the "
                       f"matching intent text in \"intent\". Do NOT invent other intents:\n{intent_lines}")
    else:
        intent_intro = (f"generate {n_qa} diverse patient-specific radiology QA pairs for {pid} ({sex}).\n"
                        "You are FREE to choose the clinical intent of each question.")
        intent_rule = ("3. Choose the intent FREELY, but keep it about the PRIMARY TARGET FINDING "
                       "(e.g. presence/description, laterality/location, severity/extent, interval "
                       "change, radiographic appearance, or a concise target-finding summary). "
                       "Put it in \"intent\"; do NOT reuse an intent.")
    example = (
        "Question: Patient 12345678 was imaged for worsening cough and fever; "
        "given a new left lower lobe opacity interpreted as pneumonia, how has "
        "the target finding changed from the prior radiograph?\n"
        "Answer: For Patient 12345678, the target finding is new left lower lobe "
        "pneumonia. The report describes a new left lower lobe opacity compared "
        "with the prior radiograph and interprets it as pneumonia."
    )
    question_len_rule = (
        f'Aim for {LEN_BOUNDS["q_min"]}-{LEN_BOUNDS["q_max"]} words.'
        if ENFORCE_LENGTH_BOUNDS else
        "Use a concise natural length; do not add filler to meet a word count."
    )
    answer_len_phrase = (
        f' ({LEN_BOUNDS["a_min"]}-{LEN_BOUNDS["a_max"]} words)'
        if ENFORCE_LENGTH_BOUNDS else
        ""
    )
    return f"""You are a radiology QA writer. Using ONLY the chest X-ray report text below,
{intent_intro} Each QA should feel like a realistic clinical-reasoning question,
NOT a short extraction question.

Primary target finding: {target_finding}
Primary target evidence span:
{a.get('evidence_section')}: {sanitize(a.get('evidence_span') or '')}

Report text (the ONLY allowed source of facts):
{unit['grounding']}

Style example (copy the STYLE only, NOT the content; use the real id {pid} and
ONLY facts from the report above):
{example}

Rules:
1. Each QUESTION weaves in 1-3 grounded facts about the PRIMARY TARGET FINDING
   ({target_finding}) before asking, and names "{pid}".
   Avoid short extraction questions like "What is the finding?" or "Is pneumothorax
   present?". {question_len_rule}
2. Each ANSWER is 2-3 sentences{answer_len_phrase} and names "{pid}": first sentence
   gives the direct answer; the second cites the supporting report evidence; an
   optional third explains the radiology significance ONLY if grounded in the report.
{intent_rule}
4. Do NOT recommend treatment or management unless the report explicitly recommends
   follow-up. You MAY explain radiology interpretation from the report only (e.g.
   worsening, stable, likely atelectasis, concerning for pneumonia, not excluded).
5. Every patient-specific QA must target the PRIMARY TARGET FINDING ({target_finding}).
   Do not ask about associated findings, pertinent negatives, devices/lines, or
   other diseases unless they are part of the target finding's wording. Use the
   primary target evidence span above as the main anchor; use other report text
   only when it directly describes or modifies {target_finding}.
6. Never output "___" or "[redacted]". question_rewrite = a paraphrase with the same
   meaning and answer, also naming "{pid}".
7. Every explanatory statement MUST be traceable to the report's wording. Do NOT use
   interpretive terms (e.g. appropriate, proper, properly placed, stable, chronic,
   acute, complication-free, immunocompromised, life support, well-positioned)
   unless that word or an equivalent fact is stated in the report. Synthesize 2-3
   report facts; add no clinical implications of your own.
8. Do NOT mention treatments, procedures, surgery, biopsy, tubes, drains, catheters,
   pacers, support devices, or follow-up management. Stay on radiographic description
   of {target_finding}.
9. {pronoun_rule}{disallowed_rule}

Return ONLY valid JSON, a list of exactly {n_qa} objects:
[{{"intent": "...", "question": "...", "question_rewrite": "...", "answer": "..."}}]
"""


def pair_pronoun_rule(anchor):
    pid = anchor["patient_ref"]
    if anchor.get("sex") in ("male", "female"):
        return f'Gendered pronouns are allowed because the report states the sex ({anchor["sex"]}).'
    return (f'The sex is NOT stated; NEVER use he/she/his/her/him for {pid}; '
            f'use only "{pid}" or "the patient".')


def build_paired_hard_prompt(forget_unit, retain_unit, n_qa, needed_qids=None):
    """V3-style HARD generation for aligned forget/retain QA.

    First pass requests all Q-slots. Retry passes can pass needed_qids so the
    model targets only missing slots instead of regenerating already-filled QIDs.
    """
    fa = forget_unit["anchor"]
    ra = retain_unit["anchor"]
    finding = fa["target_finding"]
    if finding != ra["target_finding"]:
        raise ValueError(f"paired hard prompt requires same finding, got {finding} vs {ra['target_finding']}")
    all_intents = aligned_intents_for(forget_unit, n_qa)
    all_qids = [f"Q{i + 1:02d}" for i in range(n_qa)]
    intent_by_qid = dict(zip(all_qids, all_intents))
    if needed_qids is None:
        selected_qids = all_qids
    else:
        selected_qids = []
        for qid in needed_qids:
            q = str(qid or "").strip().upper()
            if q in intent_by_qid and q not in selected_qids:
                selected_qids.append(q)
        if not selected_qids:
            selected_qids = all_qids
    n_requested = len(selected_qids)
    requested = ", ".join(selected_qids)
    return f"""You are a radiology QA writer creating a coupled paired benchmark example. You will receive TWO different chest X-ray reports for two different patients. Both patients share the same target finding: "{finding}".

Goal: generate exactly {n_requested} aligned question_id group(s), only for: {requested}. For each, produce BOTH a forget-patient QA and a retain-patient QA. The two sides must ask the same clinical aspect and use a comparable answer structure, so their answers overlap in radiology phrasing. Each answer must use ONLY its own patient's report facts. Never copy laterality, severity, location, devices, negatives, or interval change from one patient to the other.

Overlap target: for each question_id, use the same sentence order and nearly the same clinical wording on both sides, changing only the patient reference and the report-specific facts. The overlap must come from the shared finding, shared intent, and shared phrasing, not from transferring facts across reports.

[Forget patient]
Patient ref: {fa['patient_ref']}; Target finding: {fa['target_finding']}
Report text allowed for the forget answer ONLY: {forget_unit['grounding']}

[Retain patient]
Patient ref: {ra['patient_ref']}; Target finding: {ra['target_finding']}
Report text allowed for the retain answer ONLY: {retain_unit['grounding']}

Rules:
1. Each answer is a 2-3 sentence skeleton grounded in that same patient's report: sentence 1 = finding status; sentence 2 = supporting report evidence; sentence 3 (optional) = an associated finding or interval change stated in that report.
2. Do not recommend treatment or prognosis, and add no clinical implication not stated in the report.
3. Never output "___" or "[redacted]". Do not use the words "forget" or "retain" inside questions or answers.

Return ONLY valid JSON, a list of exactly {n_requested} flat objects with fields: question_id, intent, forget_question, forget_question_rewrite, forget_answer, retain_question, retain_question_rewrite, retain_answer.
"""

def paired_item_for_role(item, role, fallback_intent="", fallback_qid=""):
    nested = item.get(role) if isinstance(item, dict) else None
    nested = nested if isinstance(nested, dict) else {}
    return {
        "intent": str(item.get("intent") or item.get(f"{role}_intent")
                      or nested.get("intent") or fallback_intent or ""),
        "question": str(item.get(f"{role}_question") or item.get(f"{role}_query")
                        or nested.get("question") or nested.get("query") or "").strip(),
        "question_rewrite": str(item.get(f"{role}_question_rewrite")
                                or item.get(f"{role}_query_rewrite")
                                or nested.get("question_rewrite")
                                or nested.get("query_rewrite") or "").strip(),
        "answer": str(item.get(f"{role}_answer") or nested.get("answer") or "").strip(),
        "question_id": str(item.get("question_id") or fallback_qid or "").strip(),
    }


def build_general_prompt(finding, n_qa):
    return f"""You are a radiology educator. Write exactly {n_qa} general question-answer
pair(s) about the chest X-ray finding "{finding}" at the disease/finding level
(NO specific patient, NO names, NO "Patient" id).

Rules:
1. General radiology knowledge about how "{finding}" is described / recognized /
   characterized on chest radiographs (definition, typical appearance, location,
   distinguishing features, common associations).
2. Each answer should be a thorough 80-120 word explanation (general knowledge is
   not tied to one report, so it can be richer than patient QA).
3. No patient-specific facts, no treatment plans beyond general description.
4. Never output "___".
5. question_rewrite = a paraphrase of question with the same answer.

Return ONLY valid JSON, a list of exactly {n_qa} objects:
[{{"question": "...", "question_rewrite": "...", "answer": "..."}}]
"""


# ============================================================
# Parse + validate
# ============================================================

def strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_list(text):
    t = strip_fences(text)
    candidates = [t]
    repaired = re.sub(r",\s*([}\]])", r"\1", t)
    if repaired != t:
        candidates.append(repaired)
    s, e = t.find("["), t.rfind("]")
    if s >= 0 and e > s:
        sliced = t[s:e + 1]
        candidates.append(sliced)
        repaired_sliced = re.sub(r",\s*([}\]])", r"\1", sliced)
        if repaired_sliced != sliced:
            candidates.append(repaired_sliced)
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            v = json.loads(cand)
            return v if isinstance(v, list) else None
        except json.JSONDecodeError:
            pass
    return None


_WORD = re.compile(r"[a-z0-9]+")
PLACEHOLDER_RE = re.compile(r"_{2,}|\[redacted\]", re.I)   # de-id markers, in any form
# configurable Q/A word-length gates (main() overrides from CLI args)
LEN_BOUNDS = {"q_min": 20, "q_max": 80, "a_min": 35, "a_max": 80}
ENFORCE_LENGTH_BOUNDS = True
GENDER_RE = re.compile(r"\b(he|she|his|her|him|hers|himself|herself)\b", re.I)

# sentence-scaffolding words excluded from grounding so a *full-sentence* answer
# is scored on its clinical facts, not connective boilerplate. "redacted" too.
SCAFFOLD = {
    "redacted", "patient", "report", "reports", "reported", "state", "states",
    "stated", "show", "shows", "showed", "shown", "demonstrate", "demonstrates",
    "demonstrated", "reveal", "reveals", "revealed", "note", "noted", "notes",
    "present", "presence", "indicate", "indicates", "indicated", "finding",
    "findings", "study", "exam", "examination", "radiograph", "radiographs",
    "image", "imaging", "chest", "xray", "film", "films", "states", "the",
    "according", "describes", "described", "seen", "consistent", "evidence",
}


def content_tokens(text):
    return set(w for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in SCAFFOLD)


def validate_qa(item, unit, ground_tokens, min_ground, finding_patterns=None, support_patterns=None):
    """Return (clean_record_or_None, errors). V3-style: clinical-reasoning QA with
    length + identity + grounding gates."""
    errs = []
    if not isinstance(item, dict):
        return None, ["not_object"]
    q = str(item.get("question") or "").strip()
    a = str(item.get("answer") or "").strip()
    qr = str(item.get("question_rewrite") or "").strip() or q
    if not q or not a:
        errs.append("empty_q_or_a")
    if PLACEHOLDER_RE.search(q) or PLACEHOLDER_RE.search(a) or PLACEHOLDER_RE.search(qr):
        errs.append("placeholder_leak")
    pid = unit["anchor"]["patient_ref"]
    if pid not in q:
        errs.append("missing_patient_ref")
    if pid not in qr:                       # rewrite is used in *_rewrite.json
        errs.append("missing_patient_ref_rewrite")
    if pid not in a:                        # answer should also carry identity
        errs.append("missing_patient_ref_answer")
    # no gendered pronouns when the report does not state the patient's sex
    if unit["anchor"].get("sex") not in ("male", "female"):
        if GENDER_RE.search(q) or GENDER_RE.search(a) or GENDER_RE.search(qr):
            errs.append("gendered_pronoun")
    disallowed = unit.get("disallowed_findings") or []
    if disallowed and finding_patterns:
        leak_text = " ".join([q, qr, a])
        hits = mentioned_findings(leak_text, disallowed, finding_patterns)
        if hits:
            errs.append("opposite_finding_mention:" + ",".join(hits))
    if support_patterns:
        support_hits = mentioned_support_terms(" ".join([q, qr, a]), support_patterns)
        if support_hits:
            errs.append("support_treatment_mention:" + ",".join(support_hits))
    if ENFORCE_LENGTH_BOUNDS:
        qw, aw = len(q.split()), len(a.split())
        if not (LEN_BOUNDS["q_min"] <= qw <= LEN_BOUNDS["q_max"]):
            errs.append(f"question_len:{qw}")
        if not (LEN_BOUNDS["a_min"] <= aw <= LEN_BOUNDS["a_max"]):
            errs.append(f"answer_len:{aw}")
    # grounding: strip patient ref (subject_id absent from de-id reports) so
    # identity scaffolding does not depress the score; require >=2 grounded facts
    at = content_tokens(a.replace(pid, " "))
    inter = at & ground_tokens
    gr = len(inter) / len(at) if at else 0.0
    if gr < min_ground:
        errs.append(f"low_grounding:{gr:.2f}")
    if len(inter) < 2:
        errs.append(f"too_few_grounded_facts:{len(inter)}")
    rec = {"question": q, "question_rewrite": qr, "answer": a,
           "intent": str(item.get("intent") or ""), "grounding_ratio": round(gr, 3),
           "n_grounded_facts": len(inter)}
    return rec, errs


def validate_general(item):
    errs = []
    if not isinstance(item, dict):
        return None, ["not_object"]
    q = str(item.get("question") or "").strip()
    a = str(item.get("answer") or "").strip()
    qr = str(item.get("question_rewrite") or "").strip() or q
    if not q or not a:
        errs.append("empty_q_or_a")
    if PLACEHOLDER_RE.search(q) or PLACEHOLDER_RE.search(a) or PLACEHOLDER_RE.search(qr):
        errs.append("placeholder_leak")
    return {"question": q, "question_rewrite": qr, "answer": a}, errs


# ============================================================
# Backends (local only)
# ============================================================

def make_client(args):
    if args.backend == "hf-batched":
        return HFBatched(args)
    if args.backend == "openai":
        return OpenAIClient(args)
    raise ValueError(f"backend {args.backend} not available for generation")


class OpenAIClient:
    """OpenAI (external API) backend with concurrent requests.

    NOTE: this sends prompt text to a third-party API. Only use for data you are
    permitted to send externally. API key from --openai-api-key or OPENAI_API_KEY
    (never hardcoded). complete_batch sends prompts concurrently and preserves order.
    """

    def __init__(self, args):
        import requests
        self.requests = requests
        self.url = args.openai_base_url.rstrip("/") + "/chat/completions"
        self.model = args.openai_model
        self.key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise ValueError("OpenAI needs --openai-api-key or OPENAI_API_KEY env")
        self.concurrency = max(1, args.openai_concurrency)
        self.max_new_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.timeout = args.timeout
        self.progress_every = max(0, args.progress_every)
        self.heartbeat_sec = max(0, args.heartbeat_sec)
        self.debug_print_raw = bool(getattr(args, "debug_print_raw", False))
        self.debug_raw_max_chars = max(0, int(getattr(args, "debug_raw_max_chars", 0) or 0))

    def _debug_print_raw(self, raw, label):
        if not self.debug_print_raw:
            return
        text = "" if raw is None else str(raw)
        shown = text
        truncated = False
        if self.debug_raw_max_chars and len(text) > self.debug_raw_max_chars:
            shown = text[:self.debug_raw_max_chars]
            truncated = True
        print(f"\n[openai][raw {label}] chars={len(text)}",
              file=sys.stderr, flush=True)
        print(shown, file=sys.stderr, flush=True)
        if truncated:
            print(f"[openai][raw {label}] truncated at {self.debug_raw_max_chars} chars",
                  file=sys.stderr, flush=True)
        print(f"[openai][raw {label}] end", file=sys.stderr, flush=True)

    def _one(self, prompt, temperature):
        temp = self.temperature if temperature is None else temperature
        body = {"model": self.model,
                "messages": [{"role": "system", "content": "Return only valid JSON."},
                             {"role": "user", "content": prompt}],
                "max_tokens": self.max_new_tokens,
                "temperature": temp if (temp and temp > 0) else 0}
        if temp and temp > 0:
            body["top_p"] = self.top_p
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        last = ""
        for attempt in range(3):
            try:
                r = self.requests.post(self.url, headers=headers,
                                       data=json.dumps(body), timeout=self.timeout)
                j = r.json()
                if "error" in j:
                    last = f"OAI_ERROR:{str(j['error'])[:80]}"
                    time.sleep(2 * (attempt + 1)); continue
                return j["choices"][0]["message"].get("content") or ""
            except Exception as e:
                last = f"OAI_EXC:{type(e).__name__}"; time.sleep(2 * (attempt + 1))
        return last

    def complete_batch(self, prompts, temperature=None, debug_labels=None):
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        out = [None] * len(prompts)
        total = len(prompts)
        if total and self.progress_every:
            print(f"[openai] submit {total} request(s), concurrency={self.concurrency}",
                  file=sys.stderr, flush=True)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            pending = {ex.submit(self._one, p, temperature): i for i, p in enumerate(prompts)}
            done_count = 0
            last_heartbeat = time.time()
            while pending:
                done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if not done:
                    now = time.time()
                    if self.heartbeat_sec and now - last_heartbeat >= self.heartbeat_sec:
                        print(f"[openai] waiting... completed {done_count}/{total}; "
                              f"pending={len(pending)}",
                              file=sys.stderr, flush=True)
                        last_heartbeat = now
                    continue
                for f in done:
                    i = pending.pop(f)
                    result = f.result()
                    out[i] = result
                    done_count += 1
                    label = (debug_labels[i] if debug_labels and i < len(debug_labels)
                             else f"request={i + 1}/{total}")
                    self._debug_print_raw(result, label)
                    if isinstance(result, str) and result.startswith(("OAI_ERROR", "OAI_EXC")):
                        print(f"[openai] request {i + 1}/{total} failed: {result}",
                              file=sys.stderr, flush=True)
                    if self.progress_every and (done_count == total or done_count % self.progress_every == 0):
                        print(f"[openai] completed {done_count}/{total}",
                              file=sys.stderr, flush=True)
        return out


class HFBatched:
    def __init__(self, args):
        if not args.model_path:
            raise ValueError("--model-path required for --backend hf-batched")
        import torch
        import transformers
        self.torch = torch
        self.batch_size = max(1, args.hf_batch_size)
        self.max_new_tokens = args.max_tokens
        self.max_input_tokens = args.hf_max_input_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.progress_every = max(0, args.progress_every)
        dev = args.device
        self.device = ("cuda:0" if (dev in ("auto",) and torch.cuda.is_available())
                       else (f"cuda:{dev}" if dev.isdigit() and torch.cuda.is_available() else dev))
        dtype = {"auto": "auto", "float16": torch.float16,
                 "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
        tok = transformers.AutoTokenizer.from_pretrained(args.model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self.tok = tok
        cls = (transformers.AutoModelForImageTextToText
               if args.hf_auto_class == "image-text-to-text"
               else transformers.AutoModelForCausalLM)
        load_kw = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        # MoE models (e.g. gemma-4-26B-A4B) default to grouped_mm which is
        # unsupported on ROCm; "batched_mm" is a pure-bmm path that works.
        if args.experts_implementation:
            load_kw["experts_implementation"] = args.experts_implementation
        self.model = cls.from_pretrained(args.model_path, **load_kw)
        self.model.to(self.device)
        self.model.eval()

    def _fmt(self, prompt):
        msgs = [{"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt}]
        if getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"System: Return only valid JSON.\n\nUser:\n{prompt}\n\nAssistant:\n"

    def complete_batch(self, prompts, temperature=None, debug_labels=None):
        out = []
        total = len(prompts)
        if total and self.progress_every:
            print(f"[hf-batched] generate {total} prompt(s), batch_size={self.batch_size}",
                  file=sys.stderr, flush=True)
        for i in range(0, total, self.batch_size):
            out.extend(self._gen(prompts[i:i + self.batch_size], temperature))
            done = min(i + self.batch_size, total)
            if self.progress_every and (done == total or done % self.progress_every == 0):
                print(f"[hf-batched] completed {done}/{total}",
                      file=sys.stderr, flush=True)
        return out

    def _gen(self, prompts, temperature=None):
        temp = self.temperature if temperature is None else temperature
        texts = [self._fmt(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_input_tokens)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        kw = {"max_new_tokens": self.max_new_tokens, "pad_token_id": self.tok.pad_token_id}
        if temp and temp > 0:
            kw.update(do_sample=True, temperature=temp, top_p=self.top_p)
        else:
            kw.update(do_sample=False)
        with self.torch.inference_mode():
            gen = self.model.generate(**enc, **kw)
        gen = gen[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


# ============================================================
# Build units
# ============================================================

def build_units(pairs, report_map, coupling, reference_pairs=None):
    easy_disallowed = {}
    if coupling == "easy":
        ref = reference_pairs or pairs
        forget_findings = {p["forget"]["target_finding"] for p in ref}
        retain_findings = {p["retain"]["target_finding"] for p in ref}
        easy_disallowed = {"forget": retain_findings, "retain": forget_findings}
    units = []
    for p in pairs:
        for role in ("forget", "retain"):
            a = p[role]
            rep = report_map.get(a["study_id"])
            if rep is None:
                continue
            ground = grounding_text(rep)
            if not ground:
                continue
            units.append({
                "uid": f"{p['pair_id']}_{role}",
                "pair_id": p["pair_id"], "role": role, "coupling_level": p["coupling_level"],
                "anchor": a, "grounding": ground,
                "disallowed_findings": sorted(easy_disallowed.get(role, [])),
            })
    return units


def pair_unit_groups(pairs, units):
    by_key = {(u["pair_id"], u["role"]): u for u in units}
    groups = []
    for p in pairs:
        f = by_key.get((p["pair_id"], "forget"))
        r = by_key.get((p["pair_id"], "retain"))
        if f is not None and r is not None:
            groups.append({"pair_id": p["pair_id"], "forget": f, "retain": r})
    return groups


def general_targets(pairs, per_finding):
    findings = sorted({p[r]["target_finding"] for p in pairs for r in ("forget", "retain")})
    return [(f, per_finding) for f in findings]


def write_split_dir(split_dir, forget_recs, retain_recs, general_recs, patient_full):
    def qa(recs, q="query"):
        return [{"question": r[q], "answer": r["answer"]} for r in recs]
    write_json(os.path.join(split_dir, "forget.json"), qa(forget_recs))
    write_json(os.path.join(split_dir, "retain.json"), qa(retain_recs))
    write_json(os.path.join(split_dir, "forget_rewrite.json"), qa(forget_recs, "query_rewrite"))
    write_json(os.path.join(split_dir, "retain_rewrite.json"), qa(retain_recs, "query_rewrite"))
    write_json(os.path.join(split_dir, "general.json"), qa(general_recs))
    write_json(os.path.join(split_dir, "full.json"), qa(patient_full))


def do_merge(out_name):
    """Combine OUTPUTS/<out_name>.shardXXofNN.json into the final master + split dir."""
    shard_paths = sorted(glob.glob(os.path.join(OUTPUTS, f"{out_name}.shard*of*.json")))
    if not shard_paths:
        print(f"ERROR: no shard files matching {out_name}.shard*of*.json", file=sys.stderr)
        return 1
    forget, retain, general = [], [], []
    err = Counter()
    unit_kept_counts = {}
    deficient_units = []
    invalid_examples = defaultdict(list)
    complete_all = True
    shard_meta = []
    for sp in shard_paths:
        d = load_json(sp)
        shard_meta.append(d)
        forget += d.get("forget", [])
        retain += d.get("retain", [])
        general += d.get("general", [])
        err.update(d.get("validation_error_counts", {}))
        unit_kept_counts.update(d.get("unit_kept_counts", {}))
        deficient_units.extend(d.get("deficient_units", []))
        for key, vals in d.get("invalid_examples", {}).items():
            invalid_examples[key].extend(vals)
        complete_all = complete_all and d.get("complete", False)
    first = shard_meta[0] if shard_meta else {}
    generation_modes = sorted({d.get("generation_mode", "unit") for d in shard_meta})
    patient_full = forget + retain
    master = {
        "benchmark_type": first.get("benchmark_type", out_name),
        "merged_from": [os.path.basename(p) for p in shard_paths],
        "complete": complete_all,
        "source_pair_pool": first.get("source_pair_pool"),
        "qa_per_patient": first.get("qa_per_patient"),
        "general_per_finding": first.get("general_per_finding"),
        "generation_mode": generation_modes[0] if len(generation_modes) == 1 else generation_modes,
        "aligned_intents": any(bool(d.get("aligned_intents")) for d in shard_meta),
        "targeted_retry": any(bool(d.get("targeted_retry")) for d in shard_meta),
        "length_gate": any(bool(d.get("length_gate", True)) for d in shard_meta),
        "length_bounds": first.get("length_bounds"),
        "count": len(patient_full) + len(general),
        "forget_count": len(forget), "retain_count": len(retain), "general_count": len(general),
        "validation_error_counts": dict(err.most_common()),
        "unit_kept_counts": unit_kept_counts,
        "deficient_units": deficient_units,
        "invalid_examples": dict(invalid_examples),
        "full": patient_full, "forget": forget, "retain": retain, "general": general,
    }
    write_json(os.path.join(OUTPUTS, f"{out_name}.json"), master)
    write_split_dir(os.path.join(OUTPUTS, out_name), forget, retain, general, patient_full)
    print(f"=== MERGE [{out_name}] from {len(shard_paths)} shards ===")
    print(f"forget={len(forget)} retain={len(retain)} general={len(general)} complete={complete_all}")
    print(f"validation_error_counts: {dict(err.most_common())}")
    if deficient_units:
        print(f"deficient units: {len(deficient_units)}")
    print(f"wrote master -> {os.path.join(OUTPUTS, out_name)}.json")
    print(f"wrote split dir -> {os.path.join(OUTPUTS, out_name)}/")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Stage 30: generate grounded radiology QA")
    ap.add_argument("--coupling", choices=["easy", "hard"], default="easy")
    ap.add_argument("--reports", default=DEFAULT_REPORTS)
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    ap.add_argument("--pool", default=None, help="pair pool json (default by coupling)")
    ap.add_argument("--out-name", default=None, help="output base name (default by coupling)")
    ap.add_argument("--qa-per-patient", type=int, default=10)
    ap.add_argument("--align-intents", action="store_true",
                    help="use a fixed finding-centric intent list so forget/retain of a "
                         "pair ask the same aspects (raises hard coupling)")
    ap.add_argument("--paired-hard-generation", action="store_true",
                    help="for hard only, generate each forget/retain pair in one V3-style "
                         "joint prompt with aligned question_id groups and parallel answers")
    ap.add_argument("--general-per-finding", type=int, default=2)
    ap.add_argument("--min-grounding", type=float, default=0.45)
    ap.add_argument("--grounding-mode", choices=["drop", "warn"], default="drop",
                    help="drop: low-grounding QA are removed; warn: kept but flagged")
    ap.add_argument("--max-retries", type=int, default=4,
                    help="extra sampling passes to top up units below qa-per-patient")
    ap.add_argument("--debug-invalid-examples", type=int, default=5,
                    help="store up to N invalid examples per validation error in the master JSON; 0 disables")
    ap.add_argument("--debug-print-raw", action="store_true",
                    help="print raw model responses to stderr before parsing")
    ap.add_argument("--debug-raw-max-chars", type=int, default=0,
                    help="max raw response chars to print per item; 0 prints full")
    ap.add_argument("--retry-temperature", type=float, default=0.7)
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--backend", choices=["dry-run", "hf-batched", "openai"], default="dry-run")
    # OpenAI (external API) options
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    ap.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    ap.add_argument("--openai-api-key", default=None, help="or set OPENAI_API_KEY env")
    ap.add_argument("--openai-concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--progress-every", type=int, default=10,
                    help="print backend progress every N completed prompts; 0 disables")
    ap.add_argument("--heartbeat-sec", type=int, default=15,
                    help="while waiting for the API, print pending requests every N seconds; 0 disables")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--hf-auto-class", choices=["image-text-to-text", "causal-lm"], default="image-text-to-text")
    ap.add_argument("--experts-implementation", default=None,
                    choices=[None, "batched_mm", "grouped_mm", "sonicmoe"],
                    help="MoE experts kernel; use batched_mm for MoE on ROCm")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hf-batch-size", type=int, default=8)
    ap.add_argument("--hf-max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=2560)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split pairs across N processes (one GPU each)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--merge", action="store_true",
                    help="merge OUTPUTS/<out_name>.shard*.json -> final master + split dir")
    ap.add_argument("--resume", action="store_true",
                    help="resume a non-sharded paired-hard run by keeping existing valid QA and filling missing qids")
    ap.add_argument("--resume-from", default=None,
                    help="master JSON to resume from; default is OUTPUTS/<out_name>.json")
    ap.add_argument("--q-min", type=int, default=None, help="question min words (override)")
    ap.add_argument("--q-max", type=int, default=None)
    ap.add_argument("--a-min", type=int, default=None, help="answer min words (override)")
    ap.add_argument("--a-max", type=int, default=None)
    ap.add_argument("--no-length-gate", action="store_true",
                    help="do not prompt for or validate patient QA question/answer word counts")
    args = ap.parse_args()

    # aligned intents are a HARD-coupling feature only; never apply to easy.
    # Paired hard generation always uses aligned intents inside its pair prompt.
    use_paired_hard = args.paired_hard_generation and args.coupling == "hard"

    # Unit/easy defaults stay 20-80 / 35-80. Paired-hard questions are deliberately
    # compact and parallel, so only that mode defaults to q_min=12. CLI overrides win.
    if use_paired_hard:
        # paired parallel forget/retain QA are intrinsically terser than single-mode
        LEN_BOUNDS["q_min"] = 12
    for k, v in (("q_min", args.q_min), ("q_max", args.q_max),
                 ("a_min", args.a_min), ("a_max", args.a_max)):
        if v is not None:
            LEN_BOUNDS[k] = v
    global ENFORCE_LENGTH_BOUNDS
    ENFORCE_LENGTH_BOUNDS = not args.no_length_gate
    if ENFORCE_LENGTH_BOUNDS:
        print(f"[stage30] length bounds: {LEN_BOUNDS}", file=sys.stderr, flush=True)
    else:
        print("[stage30] length gate disabled", file=sys.stderr, flush=True)
    use_aligned = args.align_intents and args.coupling == "hard" and not use_paired_hard
    if args.paired_hard_generation and args.coupling != "hard":
        print("[stage30] WARNING: --paired-hard-generation is ignored unless --coupling hard",
              file=sys.stderr, flush=True)

    out_name = args.out_name or f"{args.coupling}_context_qa_mimic_4000"
    if args.merge:
        return do_merge(out_name)

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        print("ERROR: bad --num-shards/--shard-index", file=sys.stderr)
        return 1
    if args.resume and args.num_shards > 1:
        print("ERROR: --resume currently supports only non-sharded runs", file=sys.stderr)
        return 1
    suffix = "" if args.num_shards <= 1 else f".shard{args.shard_index:02d}of{args.num_shards:02d}"

    pool_path = args.pool or os.path.join(POOLS, f"{args.coupling}_pair_pool.json")
    master_path = os.path.join(OUTPUTS, f"{out_name}{suffix}.json")
    split_dir = os.path.join(OUTPUTS, out_name)   # split dir only written when not sharded
    resume_path = args.resume_from or master_path

    for path in (pool_path, args.reports):
        if not os.path.exists(path):
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1
    existing_master = None
    if args.resume:
        if not os.path.exists(resume_path):
            print(f"ERROR: --resume source not found: {resume_path}", file=sys.stderr)
            return 1
        existing_master = load_json(resume_path)
        print(f"[stage30] resume from {resume_path}", file=sys.stderr, flush=True)

    pool = load_json(pool_path)
    all_pairs = pool["pairs"]
    pairs = all_pairs
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]
    # general is computed from the FULL (pre-shard) pair set and produced ONLY by
    # shard 0, so it is generated exactly once and not duplicated across shards.
    gen_targets = general_targets(pairs, args.general_per_finding) if args.shard_index == 0 else []
    if args.num_shards > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_index]
    report_map = {r["study_id"]: r for r in load_json(args.reports)}
    finding_patterns = load_finding_patterns(args.taxonomy)
    support_patterns = load_support_patterns(args.taxonomy)

    units = build_units(pairs, report_map, args.coupling, reference_pairs=all_pairs)
    paired_groups = pair_unit_groups(pairs, units)

    print(f"[stage30] coupling={args.coupling} backend={args.backend} out_name={out_name}",
          file=sys.stderr, flush=True)
    print(f"[stage30] pairs={len(pairs)} patient_units={len(units)} "
          f"qa_per_patient={args.qa_per_patient} max_retries={args.max_retries}",
          file=sys.stderr, flush=True)

    # prompt preview / dry-run
    if use_paired_hard:
        preview = [{"pair": {"pair_id": g["pair_id"],
                              "coupling_level": g["forget"]["coupling_level"],
                              "forget_patient_ref": g["forget"]["anchor"]["patient_ref"],
                              "retain_patient_ref": g["retain"]["anchor"]["patient_ref"]},
                    "prompt": build_paired_hard_prompt(g["forget"], g["retain"], args.qa_per_patient)}
                   for g in paired_groups[:3]]
    else:
        preview = [{"unit": {k: u[k] for k in ("pair_id", "role", "coupling_level")},
                    "patient_ref": u["anchor"]["patient_ref"],
                    "prompt": build_patient_prompt(u, args.qa_per_patient,
                              aligned_intents=aligned_intents_for(u, args.qa_per_patient) if use_aligned else None)}
                   for u in units[:3]]
    if gen_targets:
        preview.append({"general_example": build_general_prompt(gen_targets[0][0], args.general_per_finding)})
    write_json(os.path.join(PROCESSED, f"30_prompt_preview_{args.coupling}.json"),
               {"backend": args.backend, "n_units": len(units),
                "n_paired_groups": len(paired_groups) if use_paired_hard else 0,
                "generation_mode": "paired_hard" if use_paired_hard else "unit",
                "n_general_targets": len(gen_targets), "items": preview})

    if args.backend == "dry-run":
        print(f"=== STAGE 30 [{args.coupling}] DRY-RUN ===")
        print(f"pairs={len(pairs)}  patient units={len(units)} (x{args.qa_per_patient} QA = {len(units)*args.qa_per_patient})")
        if use_paired_hard:
            print(f"paired hard groups={len(paired_groups)} (x{args.qa_per_patient} paired QA groups)")
        print(f"general targets={len(gen_targets)} (x{args.general_per_finding} = {len(gen_targets)*args.general_per_finding})")
        print(f"prompt preview -> processed/30_prompt_preview_{args.coupling}.json")
        print("rerun with --backend hf-batched --model-path ... to generate")
        return 0

    client = make_client(args)
    started = time.time()

    # ---- patient QA (retry loop tops up units below qa_per_patient) ----
    err_counts = Counter()
    invalid_examples = defaultdict(list)
    if existing_master:
        err_counts.update(existing_master.get("validation_error_counts", {}))
        for key, vals in existing_master.get("invalid_examples", {}).items():
            invalid_examples[key].extend(vals)
    FATAL = {"placeholder_leak", "empty_q_or_a", "missing_patient_ref",
             "missing_patient_ref_rewrite", "missing_patient_ref_answer",
             "question_len", "answer_len", "too_few_grounded_facts", "gendered_pronoun",
             "not_object", "opposite_finding_mention", "support_treatment_mention"}

    def clip(text, n=220):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text if len(text) <= n else text[:n - 3] + "..."

    def debug_print_raw(kind, raw, label=None):
        if not args.debug_print_raw:
            return
        if getattr(client, "debug_print_raw", False):
            return
        text = "" if raw is None else str(raw)
        shown = text
        truncated = False
        if args.debug_raw_max_chars and args.debug_raw_max_chars > 0:
            if len(text) > args.debug_raw_max_chars:
                shown = text[:args.debug_raw_max_chars]
                truncated = True
        label_part = f" {label}" if label else ""
        print(f"\n[stage30][raw {kind}{label_part}] chars={len(text)}",
              file=sys.stderr, flush=True)
        print(shown, file=sys.stderr, flush=True)
        if truncated:
            print(f"[stage30][raw {kind}{label_part}] truncated at "
                  f"{args.debug_raw_max_chars} chars", file=sys.stderr, flush=True)
        print(f"[stage30][raw {kind}{label_part}] end", file=sys.stderr, flush=True)

    def remember_invalid(kind, errs, unit=None, item=None, raw=None, finding=None, attempt_no=None):
        if args.debug_invalid_examples <= 0:
            return
        keys = [e.split(":")[0] for e in (errs or [kind])]
        for key in keys:
            if len(invalid_examples[key]) >= args.debug_invalid_examples:
                continue
            ex = {"kind": kind}
            if attempt_no is not None:
                ex["attempt"] = attempt_no
            if unit is not None:
                ex.update({
                    "uid": unit["uid"],
                    "pair_id": unit["pair_id"],
                    "role": unit["role"],
                    "patient_ref": unit["anchor"]["patient_ref"],
                    "study_id": unit["anchor"]["study_id"],
                    "target_finding": unit["anchor"]["target_finding"],
                })
            if finding is not None:
                ex["target_finding"] = finding
            if errs:
                ex["errors"] = errs
            if isinstance(item, dict):
                ex["question"] = clip(item.get("question"))
                ex["question_rewrite"] = clip(item.get("question_rewrite"))
                ex["answer"] = clip(item.get("answer"))
                if item.get("intent") is not None:
                    ex["intent"] = clip(item.get("intent"), 120)
            elif item is not None:
                ex["item_preview"] = clip(item)
            if raw is not None:
                ex["raw_preview"] = clip(raw, 500)
            invalid_examples[key].append(ex)

    def is_fatal(errs):
        for e in errs:
            p = e.split(":")[0]
            if p in FATAL:
                return True
            if p == "low_grounding" and args.grounding_mode == "drop":
                return True
        return False

    kept = defaultdict(list)     # uid -> [(unit, rec)]
    seen_q = defaultdict(set)     # uid -> normalized questions (dedup)
    unit_by_uid = {u["uid"]: u for u in units}

    def restore_paired_from_master(master, expected_qids, filled_qids):
        group_by_pair = {g["pair_id"]: g for g in paired_groups}
        by_slot = defaultdict(dict)
        skipped = 0
        for row in master.get("full", []):
            pair_id = row.get("pair_id")
            role = row.get("patient_role")
            qid = str(row.get("question_id") or "").strip().upper()
            if pair_id not in group_by_pair or role not in ("forget", "retain") or qid not in expected_qids:
                skipped += 1
                continue
            by_slot[(pair_id, qid)][role] = row

        restored_slots = 0
        incomplete_slots = 0
        for (pair_id, qid), roles in sorted(by_slot.items()):
            if "forget" not in roles or "retain" not in roles:
                incomplete_slots += 1
                continue
            slot_stale = False
            for role in ("forget", "retain"):
                row = roles[role]
                anchor = group_by_pair[pair_id][role]["anchor"]
                if (row.get("subject_id") != anchor["subject_id"]
                        or row.get("study_id") != anchor["study_id"]
                        or row.get("target_finding") != anchor["target_finding"]):
                    slot_stale = True
                    break
            if slot_stale:
                incomplete_slots += 1
                continue
            for role in ("forget", "retain"):
                row = roles[role]
                unit = group_by_pair[pair_id][role]
                kept[unit["uid"]].append({
                    "intent": row.get("clinical_intent", ""),
                    "question": row.get("query", ""),
                    "question_rewrite": row.get("query_rewrite", "") or row.get("query", ""),
                    "answer": row.get("answer", ""),
                    "grounding_ratio": row.get("grounding_ratio", 0.0),
                    "question_id": qid,
                    "paired_generation": row.get("generation_mode") == "paired_hard",
                })
            filled_qids[pair_id].add(qid)
            restored_slots += 1

        for uid in list(kept):
            kept[uid].sort(key=lambda r: r.get("question_id", ""))
        return restored_slots, incomplete_slots, skipped

    def restore_unit_from_master(master):
        restored = 0
        skipped = 0
        for row in master.get("full", []):
            pair_id = row.get("pair_id")
            role = row.get("patient_role")
            uid = f"{pair_id}_{role}"
            if uid not in unit_by_uid:
                skipped += 1
                continue
            unit = unit_by_uid[uid]
            anchor = unit["anchor"]
            if (row.get("subject_id") != anchor["subject_id"]
                    or row.get("study_id") != anchor["study_id"]
                    or row.get("target_finding") != anchor["target_finding"]):
                skipped += 1
                continue
            item = {
                "intent": row.get("clinical_intent", ""),
                "question": row.get("query", ""),
                "question_rewrite": row.get("query_rewrite", "") or row.get("query", ""),
                "answer": row.get("answer", ""),
            }
            gt = content_tokens(unit["grounding"])
            rec, errs = validate_qa(item, unit, gt, args.min_grounding,
                                    finding_patterns, support_patterns)
            for e in errs:
                err_counts[e.split(":")[0]] += 1
            if rec is None or is_fatal(errs):
                skipped += 1
                remember_invalid("resume_validation_drop", errs, unit=unit, item=item)
                continue
            if row.get("question_id"):
                rec["question_id"] = row["question_id"]
            kept[uid].append(rec)
            qn = re.sub(r"\s+", " ", rec["question"].lower()).strip()
            if qn:
                seen_q[uid].add(qn)
            restored += 1
        for uid in list(kept):
            kept[uid] = kept[uid][:args.qa_per_patient]
        return restored, skipped

    attempt = 0
    if use_paired_hard:
        expected_qids = [f"Q{i + 1:02d}" for i in range(args.qa_per_patient)]
        filled_qids = defaultdict(set)   # pair_id -> set of filled Q-slots (one rec each)
        if existing_master:
            restored, incomplete, skipped = restore_paired_from_master(existing_master, expected_qids, filled_qids)
            print(f"[stage30] resume restored paired_slots={restored} "
                  f"patient_qa={2 * restored} incomplete_slots={incomplete} skipped_records={skipped}",
                  file=sys.stderr, flush=True)
        pending_pairs = [g for g in paired_groups
                         if len(filled_qids[g["pair_id"]]) < args.qa_per_patient]
        while pending_pairs and attempt <= args.max_retries:
            jobs = []
            missing_counter = Counter()
            for g in pending_pairs:
                pid = g["pair_id"]
                missing = [qid for qid in expected_qids if qid not in filled_qids[pid]]
                if missing:
                    jobs.append((g, missing))
                    missing_counter.update(missing)
            if not jobs:
                break
            temp = None if attempt == 0 else args.retry_temperature
            print(f"[stage30] paired hard pass {attempt + 1}/{args.max_retries + 1}: "
                  f"pending_pairs={len(jobs)} missing_slots={sum(missing_counter.values())} "
                  f"temperature={temp if temp is not None else args.temperature}",
                  file=sys.stderr, flush=True)
            if attempt > 0 and missing_counter:
                print(f"[stage30] targeted retry missing qids: {dict(missing_counter.most_common())}",
                      file=sys.stderr, flush=True)
            prompts = [build_paired_hard_prompt(g["forget"], g["retain"], args.qa_per_patient,
                                                needed_qids=missing)
                       for g, missing in jobs]
            labels = [f"paired attempt={attempt + 1} pair_id={g['pair_id']} "
                      f"qids={','.join(missing)}"
                      for g, missing in jobs]
            raws = client.complete_batch(prompts, temperature=temp, debug_labels=labels)
            nxt = []
            for (g, needed_qids), raw in zip(jobs, raws):
                fu, ru = g["forget"], g["retain"]
                pid = g["pair_id"]
                needed_set = set(needed_qids)
                debug_print_raw("paired", raw,
                                f"attempt={attempt + 1} pair_id={pid} qids={','.join(needed_qids)}")
                items = parse_list(raw) or []
                if not items:
                    err_counts["paired_parse_fail"] += 1
                    remember_invalid("paired_parse_fail", ["paired_parse_fail"], unit=fu,
                                     raw=raw, attempt_no=attempt + 1)
                fgt = content_tokens(fu["grounding"])
                rgt = content_tokens(ru["grounding"])
                expected_intents = aligned_intents_for(fu, args.qa_per_patient)
                intent_by_qid = dict(zip(expected_qids, expected_intents))
                for idx, item in enumerate(items):
                    if len(filled_qids[pid]) >= args.qa_per_patient:
                        break
                    if not isinstance(item, dict):
                        err_counts["not_object"] += 1
                        remember_invalid("paired_not_object", ["not_object"], unit=fu,
                                         item=item, attempt_no=attempt + 1)
                        continue

                    qid = str(item.get("question_id") or "").strip().upper()
                    if qid not in needed_set:
                        if not qid or qid not in expected_qids:
                            if idx < len(needed_qids):
                                err_counts["remapped_qid"] += 1
                                qid = needed_qids[idx]
                            else:
                                err_counts["unexpected_qid"] += 1
                                continue
                        else:
                            # The model returned an already-filled or non-requested Q-slot.
                            err_counts["duplicate_qid"] += 1
                            continue
                    if qid in filled_qids[pid]:
                        err_counts["duplicate_qid"] += 1
                        continue

                    fallback_intent = intent_by_qid[qid]
                    role_payloads = {
                        "forget": paired_item_for_role(item, "forget", fallback_intent, qid),
                        "retain": paired_item_for_role(item, "retain", fallback_intent, qid),
                    }
                    role_units = {"forget": fu, "retain": ru}
                    role_gt = {"forget": fgt, "retain": rgt}
                    role_recs = {}
                    drop_pair = False
                    for role in ("forget", "retain"):
                        rec, errs = validate_qa(role_payloads[role], role_units[role],
                                                role_gt[role], args.min_grounding,
                                                finding_patterns, support_patterns)
                        for e in errs:
                            err_counts[e.split(":")[0]] += 1
                        if rec is None or is_fatal(errs):
                            remember_invalid("paired_validation_drop", errs, unit=role_units[role],
                                             item=role_payloads[role], attempt_no=attempt + 1)
                            drop_pair = True
                        else:
                            rec["question_id"] = qid
                            rec["paired_generation"] = True
                            role_recs[role] = rec
                    if drop_pair:
                        continue
                    filled_qids[pid].add(qid)
                    kept[fu["uid"]].append(role_recs["forget"])
                    kept[ru["uid"]].append(role_recs["retain"])

                missing_after = [qid for qid in expected_qids if qid not in filled_qids[pid]]
                if missing_after:
                    nxt.append(g)
            kept_total = sum(len(v) for v in kept.values())
            remaining_counter = Counter(
                qid for g in nxt for qid in expected_qids if qid not in filled_qids[g["pair_id"]]
            )
            print(f"[stage30] paired hard pass {attempt + 1} done: kept_qa={kept_total} "
                  f"still_deficient_pairs={len(nxt)} missing_slots={sum(remaining_counter.values())}",
                  file=sys.stderr, flush=True)
            if remaining_counter:
                print(f"[stage30] remaining missing qids: {dict(remaining_counter.most_common())}",
                      file=sys.stderr, flush=True)
            if err_counts:
                print(f"[stage30] validation top errors so far: {dict(err_counts.most_common(8))}",
                      file=sys.stderr, flush=True)
            pending_pairs = nxt
            attempt += 1
        # order each unit's kept QA by Q-slot so forget Qi <-> retain Qi line up
        for uid in list(kept):
            kept[uid].sort(key=lambda r: r.get("question_id", ""))
    else:
        if existing_master:
            restored, skipped = restore_unit_from_master(existing_master)
            print(f"[stage30] resume restored unit_qa={restored} skipped_records={skipped}",
                  file=sys.stderr, flush=True)
        pending = [u for u in units if len(kept[u["uid"]]) < args.qa_per_patient]
        while pending and attempt <= args.max_retries:
            temp = None if attempt == 0 else args.retry_temperature
            print(f"[stage30] patient pass {attempt + 1}/{args.max_retries + 1}: "
                  f"pending_units={len(pending)} temperature={temp if temp is not None else args.temperature}",
                  file=sys.stderr, flush=True)
            prompts = [build_patient_prompt(u, args.qa_per_patient,
                aligned_intents=aligned_intents_for(u, args.qa_per_patient) if use_aligned else None)
                for u in pending]
            labels = [f"patient attempt={attempt + 1} uid={u['uid']}" for u in pending]
            raws = client.complete_batch(prompts, temperature=temp, debug_labels=labels)
            nxt = []
            for u, raw in zip(pending, raws):
                uid = u["uid"]
                debug_print_raw("patient", raw, f"attempt={attempt + 1} uid={uid}")
                items = parse_list(raw) or []
                if not items:
                    err_counts["patient_parse_fail"] += 1
                    remember_invalid("patient_parse_fail", ["patient_parse_fail"], unit=u,
                                     raw=raw, attempt_no=attempt + 1)
                gt = content_tokens(u["grounding"])
                for item in items:
                    if len(kept[uid]) >= args.qa_per_patient:
                        break
                    rec, errs = validate_qa(item, u, gt, args.min_grounding,
                                            finding_patterns, support_patterns)
                    for e in errs:
                        err_counts[e.split(":")[0]] += 1
                    if rec is None or is_fatal(errs):
                        remember_invalid("patient_validation_drop", errs, unit=u, item=item,
                                         attempt_no=attempt + 1)
                        continue
                    qn = re.sub(r"\s+", " ", rec["question"].lower()).strip()
                    if qn in seen_q[uid]:
                        err_counts["duplicate_question"] += 1
                        remember_invalid("patient_duplicate_drop", ["duplicate_question"], unit=u,
                                         item=item, attempt_no=attempt + 1)
                        continue
                    seen_q[uid].add(qn)
                    kept[uid].append(rec)
                if len(kept[uid]) < args.qa_per_patient:
                    nxt.append(u)
            kept_total = sum(len(v) for v in kept.values())
            print(f"[stage30] patient pass {attempt + 1} done: kept_qa={kept_total} "
                  f"still_deficient_units={len(nxt)}",
                  file=sys.stderr, flush=True)
            if err_counts:
                print(f"[stage30] validation top errors so far: {dict(err_counts.most_common(8))}",
                      file=sys.stderr, flush=True)
            pending = nxt
            attempt += 1

    forget_recs, retain_recs = [], []
    for u in units:
        a = u["anchor"]
        for i, rec in enumerate(kept[u["uid"]], start=1):
            qid = rec.get("question_id", f"Q{i:02d}")
            full = {
                "qa_id": f"{u['pair_id']}_{u['role']}_{qid}",
                "pair_id": u["pair_id"], "question_type": "patient_specific",
                "coupling_level": u["coupling_level"], "patient_role": u["role"],
                "subject_id": a["subject_id"], "study_id": a["study_id"],
                "patient_ref": a["patient_ref"], "sex": a["sex"],
                "target_finding": a["target_finding"], "target_family": a["target_family"],
                "evidence_section": a["evidence_section"], "evidence_span": a["evidence_span"],
                "clinical_intent": rec["intent"], "question_id": qid,
                "generation_mode": "paired_hard" if rec.get("paired_generation") else "unit",
                "query": rec["question"], "query_rewrite": rec["question_rewrite"],
                "answer": rec["answer"], "grounding_ratio": rec["grounding_ratio"],
            }
            (forget_recs if u["role"] == "forget" else retain_recs).append(full)

    deficient = [uid for uid, recs in kept.items() if len(recs) < args.qa_per_patient]
    deficient_units = [{
        "uid": uid,
        "pair_id": unit_by_uid[uid]["pair_id"],
        "role": unit_by_uid[uid]["role"],
        "patient_ref": unit_by_uid[uid]["anchor"]["patient_ref"],
        "study_id": unit_by_uid[uid]["anchor"]["study_id"],
        "target_finding": unit_by_uid[uid]["anchor"]["target_finding"],
        "kept": len(kept[uid]),
        "target": args.qa_per_patient,
    } for uid in deficient]

    # ---- general QA ----
    general_recs = list(existing_master.get("general", [])) if existing_master else []
    if existing_master and general_recs:
        print(f"[stage30] resume reused general={len(general_recs)}", file=sys.stderr, flush=True)
    else:
        gprompts = [build_general_prompt(f, n) for f, n in gen_targets]
        glabels = [f"general finding={finding}" for finding, _ in gen_targets]
        print(f"[stage30] general prompts={len(gprompts)}", file=sys.stderr, flush=True)
        graws = client.complete_batch(gprompts, debug_labels=glabels)
        for (finding, n), raw in zip(gen_targets, graws):
            debug_print_raw("general", raw, f"finding={finding}")
            items = parse_list(raw) or []
            if not items:
                err_counts["general_parse_fail"] += 1
                remember_invalid("general_parse_fail", ["general_parse_fail"], raw=raw, finding=finding)
            for k, item in enumerate(items[:n]):
                rec, errs = validate_general(item)
                for e in errs:
                    err_counts[e.split(":")[0]] += 1
                if rec is None or "placeholder_leak" in errs or "empty_q_or_a" in errs:
                    remember_invalid("general_validation_drop", errs, item=item, finding=finding)
                    continue
                general_recs.append({
                    "qa_id": f"GEN_{re.sub(r'[^A-Za-z0-9]+','_',finding)}_{k+1:02d}",
                    "question_type": "general", "target_finding": finding,
                    "query": rec["question"], "query_rewrite": rec["question_rewrite"],
                    "answer": rec["answer"],
                })

    patient_full = forget_recs + retain_recs
    _exp_f = sum(1 for u in units if u["role"] == "forget") * args.qa_per_patient
    _exp_r = sum(1 for u in units if u["role"] == "retain") * args.qa_per_patient
    # ---- master file (rich) ----
    master = {
        "timestamp_note": "stamp after run",
        "complete": len(forget_recs) >= _exp_f and len(retain_recs) >= _exp_r,
        "benchmark_type": f"{args.coupling}_context_qa_mimic",
        "source_pair_pool": pool_path, "qa_per_patient": args.qa_per_patient,
        "general_per_finding": args.general_per_finding,
        "generation_mode": "paired_hard" if use_paired_hard else "unit",
        "aligned_intents": bool(use_aligned or use_paired_hard),
        "targeted_retry": bool(use_paired_hard),
        "resumed_from": resume_path if existing_master else None,
        "length_gate": bool(ENFORCE_LENGTH_BOUNDS),
        "length_bounds": dict(LEN_BOUNDS),
        "count": len(patient_full) + len(general_recs),
        "forget_count": len(forget_recs), "retain_count": len(retain_recs),
        "general_count": len(general_recs),
        "validation_error_counts": dict(err_counts.most_common()),
        "unit_kept_counts": {u["uid"]: len(kept[u["uid"]]) for u in units},
        "deficient_units": deficient_units,
        "invalid_examples": dict(invalid_examples),
        "elapsed_sec": round(time.time() - started, 1),
        "full": patient_full, "forget": forget_recs, "retain": retain_recs,
        "general": general_recs,
    }
    write_json(master_path, master)

    # split dir is written only for a non-sharded run; sharded runs write a
    # master shard and rely on `--merge` to assemble the final split dir.
    if args.num_shards <= 1:
        write_split_dir(split_dir, forget_recs, retain_recs, general_recs, patient_full)
        print(f"wrote split dir -> {split_dir}/")
    else:
        print(f"(sharded) wrote master shard -> {master_path}; run --merge when all shards done")

    n_forget_units = sum(1 for u in units if u["role"] == "forget")
    n_retain_units = sum(1 for u in units if u["role"] == "retain")
    exp_f = n_forget_units * args.qa_per_patient
    exp_r = n_retain_units * args.qa_per_patient

    print(f"=== STAGE 30 [{args.coupling}] ===")
    print(f"forget={len(forget_recs)}/{exp_f}  retain={len(retain_recs)}/{exp_r}  "
          f"general={len(general_recs)}  (elapsed {master['elapsed_sec']}s)")
    print(f"validation_error_counts: {dict(err_counts.most_common())}")
    print(f"deficient units (<{args.qa_per_patient} QA after {args.max_retries} retries): {len(deficient)}")
    print(f"wrote master -> {master_path}")

    short = len(forget_recs) < exp_f or len(retain_recs) < exp_r
    if n_forget_units < len(pairs) or n_retain_units < len(pairs):
        print(f"WARNING: {len(pairs)-n_forget_units} forget / {len(pairs)-n_retain_units} "
              f"retain pairs had no usable report (dropped at build)")
    if short:
        print(f"COUNT GATE FAILED: forget/retain below target; "
              f"increase --max-retries or inspect deficient units")
        return 2
    print("COUNT GATE PASSED: forget == retain == target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
