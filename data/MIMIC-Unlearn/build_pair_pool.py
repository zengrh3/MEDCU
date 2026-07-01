"""
Stage 20 - Build pair pool (NO LLM)
===================================
Turns audited evidence anchors into forget/retain pairs for the EASY
(zero-overlap) and HARD (same-finding) benchmarks. Grounded on real MIMIC
anchors; mirrors MedicalGPT_V3's pair-pool design.

IDENTITY: MIMIC reports are de-identified (names are `___`), so there are NO
names to use. The patient identity is the REAL de-identified `subject_id`
(patient_ref = "Patient <subject_id>"). Sex is the REAL sex stated in the
report (___F / ___M / man / woman) when present, else null. Nothing fabricated.

Pairing unit = a study-level anchor (subject_id + study_id + target_finding +
evidence span + attributes), one representative anchor per subject (its rarest
finding) so distinct patients are maximized.

EASY (coupling=easy)  -> pools/easy_pair_pool.json
  forget & retain anchors satisfy finding_taxonomy.audit.easy_overlap_rules:
    different fine finding, different family, no blocked_with edge, distinct
    subjects. Identity disjointness is automatic (different subject_id).
  forget & retain findings balanced round-robin for diversity.

HARD (coupling=hard)  -> pools/hard_pair_pool.json   (reads easy pool)
  forget anchor byte-identical to easy forget (variable control). retain is the
  SAME fine finding, DIFFERENT subject, and (l2) DIFFERENT attributes.

Both write pools/diversity_report.json and run a pool audit. NO LLM.

Usage:
  python 20_build_pair_pool.py --coupling easy
  python 20_build_pair_pool.py --coupling hard
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MIMIC_ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(MIMIC_ROOT, "config")
PROCESSED = os.path.join(MIMIC_ROOT, "processed")
POOLS = os.path.join(MIMIC_ROOT, "pools")

DEFAULT_ANCHORS = os.path.join(PROCESSED, "evidence_anchors.json")
DEFAULT_REPORTS = os.path.join(PROCESSED, "reports.json")
DEFAULT_TAXONOMY = os.path.join(CONFIG, "finding_taxonomy.yaml")
DEFAULT_EASY = os.path.join(POOLS, "easy_pair_pool.json")
DEFAULT_HARD = os.path.join(POOLS, "hard_pair_pool.json")
DEFAULT_DIVERSITY = os.path.join(POOLS, "diversity_report.json")

ATTR_AXES = ("severity", "laterality", "location", "trend")

# real sex as stated in the de-identified report (___F / ___M / man / woman).
# word boundaries keep "male" out of "female" and "man" out of "woman".
SEX_F_RE = re.compile(r"___\s*F\b|\bfemale\b|\bwoman\b", re.I)
SEX_M_RE = re.compile(r"___\s*M\b|\bmale\b|\bman\b", re.I)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_taxonomy(path):
    with open(path, encoding="utf-8") as fh:
        tax = yaml.safe_load(fh)
    findings = tax.get("findings") or {}
    family_of = {f: c.get("family") for f, c in findings.items()}
    blocked_with = {f: set(c.get("blocked_with") or []) for f, c in findings.items()}
    rules = (tax.get("audit") or {}).get("easy_overlap_rules") or {}
    device_patterns = (findings.get("Support Devices") or {}).get("prescreen_patterns") or []
    return family_of, blocked_with, rules, device_patterns


def extract_sex(report):
    """Real sex stated in the report (INDICATION first, then whole report)."""
    sections = report.get("sections", {}) or {}
    text = sections.get("INDICATION", "") or ""
    if not text:
        text = " ".join(v for v in sections.values() if isinstance(v, str))
    is_f = bool(SEX_F_RE.search(text))
    is_m = bool(SEX_M_RE.search(text))
    if is_f and not is_m:
        return "female"
    if is_m and not is_f:
        return "male"
    return None  # absent or conflicting -> unknown


def build_sex_map(reports, study_ids):
    """study_id -> real report sex, with subject-level backfill.

    Some MIMIC-CXR reports omit sex in the selected study but another study for
    the same de-identified subject states it. Backfill only when all observed
    sex mentions for that subject agree; conflicting subjects remain unknown.
    """
    by_subject = defaultdict(set)
    direct = {}
    for r in reports:
        sex = extract_sex(r)
        if sex:
            by_subject[r["subject_id"]].add(sex)
        if r["study_id"] in study_ids:
            direct[r["study_id"]] = sex
    subject_sex = {s: next(iter(vals)) for s, vals in by_subject.items() if len(vals) == 1}
    out = {}
    for r in reports:
        if r["study_id"] in study_ids:
            out[r["study_id"]] = direct.get(r["study_id"]) or subject_sex.get(r["subject_id"])
    return out


def compile_phrase_re(phrases):
    terms = [p for p in phrases if p]
    if not terms:
        return None
    parts = [r"(?<![A-Za-z0-9])" + re.escape(p) + r"(?![A-Za-z0-9])" for p in terms]
    return re.compile("|".join(parts), re.I)


def report_text(report):
    sections = (report or {}).get("sections", {}) or {}
    return " ".join(v for v in sections.values() if isinstance(v, str))


def finding_abbr(finding):
    return "".join(w[0] for w in finding.split())[:4].upper()


def attr_tuple(anchor):
    return tuple(anchor.get(ax) for ax in ATTR_AXES)


def subject_profiles(anchors):
    """Patient-level positive profile: subject -> {findings}, subject -> {families}.
    Built from ALL anchors (a subject may have several positive findings across
    studies), so overlap is judged at the PATIENT level, not the chosen anchor."""
    find = defaultdict(set)
    fam = defaultdict(set)
    for a in anchors:
        find[a["subject_id"]].add(a["target_finding"])
        fam[a["subject_id"]].add(a["target_family"])
    return find, fam


def clean_anchors(anchors):
    """Selectable anchors: evidence span must not leak a `___` de-id placeholder."""
    out = []
    for a in anchors:
        if "___" in (a.get("evidence_span") or ""):
            continue
        if any("___" in (s.get("text") or "") for s in (a.get("all_evidence_spans") or [])):
            continue
        out.append(a)
    return out


def representative_anchors(anchors):
    """One anchor per subject: its rarest finding (smallest global count)."""
    rarity = Counter(a["target_finding"] for a in anchors)
    best = {}
    for a in anchors:
        s = a["subject_id"]
        if s not in best or rarity[a["target_finding"]] < rarity[best[s]["target_finding"]]:
            best[s] = a
    return list(best.values())


def balanced_forget_quotas(by_finding, n_pairs):
    """Balanced quotas that reserve one same-finding hard-retain subject.

    If easy consumes every eligible subject for a rare finding, hard cannot later
    build same-finding retain pairs under the same cleanliness filters. Cap easy
    usage at floor(eligible_subjects / 2) per forget finding, then distribute the
    requested pair count as evenly as possible under those caps.
    """
    caps = {f: len(bucket) // 2 for f, bucket in by_finding.items()}
    quotas = {f: 0 for f in by_finding}
    for _ in range(n_pairs):
        candidates = [f for f in quotas if quotas[f] < caps[f]]
        if not candidates:
            break
        chosen = min(candidates, key=lambda f: (quotas[f], -caps[f], f))
        quotas[chosen] += 1
    return quotas


def profiles_compatible(sa, sb, subj_find, subj_fam, blocked_with):
    """Patient-level easy compatibility between two subjects' FULL profiles."""
    if sa == sb:
        return False
    if subj_find[sa] & subj_find[sb]:          # any shared fine finding
        return False
    if subj_fam[sa] & subj_fam[sb]:            # any shared family
        return False
    for f in subj_find[sa]:                     # blocked_with edge either way
        if blocked_with.get(f, set()) & subj_fam[sb]:
            return False
    for f in subj_find[sb]:
        if blocked_with.get(f, set()) & subj_fam[sa]:
            return False
    return True


def anchor_view(anchor, sex_of):
    sid = anchor["subject_id"]
    return {
        "subject_id": sid,
        "study_id": anchor["study_id"],
        "patient_ref": f"Patient {sid}",          # identity = real de-identified ID
        "sex": sex_of.get(anchor["study_id"]),     # real sex from report, or null
        "target_finding": anchor["target_finding"],
        "target_family": anchor["target_family"],
        "evidence_section": anchor["evidence_section"],
        "evidence_span": anchor["evidence_span"],
        "all_evidence_spans": anchor.get("all_evidence_spans") or [],
        "severity": anchor.get("severity"),
        "laterality": anchor.get("laterality"),
        "location": anchor.get("location"),
        "trend": anchor.get("trend"),
        "attributes": anchor.get("attributes") or {},
    }


# ============================================================
# EASY
# ============================================================

def build_easy(anchors, subj_find, subj_fam, blocked_with, sex_of, n_pairs, seed):
    rng = random.Random(seed)
    # selection pool: clean anchors only (no `___`), one representative per subject
    reps = representative_anchors(clean_anchors(anchors))
    by_finding = defaultdict(list)
    for a in reps:
        by_finding[a["target_finding"]].append(a)
    for bucket in by_finding.values():
        rng.shuffle(bucket)
    findings = sorted(by_finding, key=lambda f: len(by_finding[f]))  # rarest first

    pairs = []
    used = set()
    retain_used = Counter()        # balance retain findings (least used first)
    deferred = defaultdict(list)   # forget anchors set aside (no partner found yet)
    fi = 0
    stalls = 0
    max_stall = len(findings) * 4
    while len(pairs) < n_pairs and stalls < max_stall:
        f_forget = findings[fi % len(findings)]
        fi += 1
        # pop an unused clean forget anchor
        f_anchor = None
        while by_finding[f_forget]:
            cand = by_finding[f_forget].pop()
            if cand["subject_id"] not in used:
                f_anchor = cand
                break
        if f_anchor is None:
            stalls += 1
            continue
        fs = f_anchor["subject_id"]
        # retain findings: least-used-as-retain first, random tiebreak (diversify combos)
        retain_candidates = list(findings)
        rng.shuffle(retain_candidates)
        retain_candidates.sort(key=lambda f: retain_used[f])
        r_anchor = chosen_rf = None
        for f_retain in retain_candidates:
            for cand in by_finding[f_retain]:
                cs = cand["subject_id"]
                if cs in used or cs == fs:
                    continue
                if profiles_compatible(fs, cs, subj_find, subj_fam, blocked_with):
                    r_anchor, chosen_rf = cand, f_retain
                    break
            if r_anchor is not None:
                break
        if r_anchor is None:
            deferred[f_forget].append(f_anchor)  # set aside; don't re-pick immediately
            stalls += 1
            continue
        stalls = 0
        used.add(fs)
        used.add(r_anchor["subject_id"])
        retain_used[chosen_rf] += 1
        pairs.append({
            "pair_id": f"EASY_{finding_abbr(f_anchor['target_finding'])}_"
                       f"{finding_abbr(r_anchor['target_finding'])}_{len(pairs)+1:04d}",
            "coupling_level": "easy_independent",
            "forget": anchor_view(f_anchor, sex_of),
            "retain": anchor_view(r_anchor, sex_of),
        })
    return pairs


# ============================================================
# EASY (concept-disjoint): forget/retain findings are GLOBALLY disjoint
# ============================================================
# Default strict split: forget/retain are globally finding/family-disjoint and
# have no taxonomy blocked_with edge between any cross-side finding pair.
DEFAULT_FORGET_FINDINGS = [
    "Consolidation", "Lung Opacity", "Pneumonia", "Lung Lesion", "Fracture",
]
DEFAULT_RETAIN_FINDINGS = [
    "Cardiomegaly", "Enlarged Cardiomediastinum",
    "Pleural Effusion", "Pneumothorax", "Pleural Other",
]


def build_easy_concept_disjoint(anchors, subj_find, subj_fam, blocked_with, sex_of,
                                n_pairs, seed, forget_findings, retain_findings,
                                report_by_study=None, support_device_re=None,
                                exclude_support_devices=True,
                                require_known_sex=True):
    """Global concept-disjoint easy: the forget set and retain set draw from
    DISJOINT finding (and family) groups, so a retain-only `gold` model never
    sees the forget concepts. A subject is eligible for a side only if its FULL
    patient profile (across all studies) is a subset of that side's findings.
    The final pairing still runs `profiles_compatible`, so cross-family clinical
    leakage edges such as Edema<->cardiomediastinal/pleural are blocked too.
    By default, studies with support devices are excluded from easy so treatment
    context does not become a shared shortcut. Anchors with unknown sex after
    subject-level backfill are also excluded to keep demographics auditable."""
    rng = random.Random(seed)
    fset, rset = set(forget_findings), set(retain_findings)
    assert not (fset & rset), f"forget/retain findings overlap: {fset & rset}"

    # Subjects whose entire profile lives on exactly one side.
    forget_subj = {s for s, fs in subj_find.items() if fs and fs <= fset}
    retain_subj = {s for s, fs in subj_find.items() if fs and fs <= rset}

    clean = clean_anchors(anchors)
    if exclude_support_devices and support_device_re is not None and report_by_study is not None:
        clean = [
            a for a in clean
            if not support_device_re.search(report_text(report_by_study.get(a["study_id"])))
        ]
    if require_known_sex:
        clean = [a for a in clean if sex_of.get(a["study_id"]) is not None]
    f_reps = representative_anchors([a for a in clean
                                    if a["subject_id"] in forget_subj
                                    and a["target_finding"] in fset])
    r_reps = representative_anchors([a for a in clean
                                    if a["subject_id"] in retain_subj
                                    and a["target_finding"] in rset])

    by_f = defaultdict(list)
    for a in f_reps:
        by_f[a["target_finding"]].append(a)
    by_r = defaultdict(list)
    for a in r_reps:
        by_r[a["target_finding"]].append(a)
    for bucket in list(by_f.values()) + list(by_r.values()):
        rng.shuffle(bucket)

    # Cycle forget findings under quotas that reserve enough same-finding
    # subjects for a strict hard pool built from this easy pool.
    forget_quota = balanced_forget_quotas(by_f, n_pairs)
    f_order = sorted((f for f in by_f if forget_quota.get(f, 0) > 0),
                     key=lambda f: (forget_quota[f], len(by_f[f]), f))
    forget_used = Counter()
    pairs, used_f, used_r = [], set(), set()
    retain_used = Counter()
    fi = stalls = 0
    max_stall = max(64, len(f_order) * 16)
    while len(pairs) < n_pairs and stalls < max_stall:
        if not f_order:
            break
        f_finding = f_order[fi % len(f_order)]
        fi += 1
        if forget_used[f_finding] >= forget_quota[f_finding]:
            if all(forget_used[f] >= forget_quota[f] for f in f_order):
                break
            stalls += 1
            continue
        f_anchor = None
        while by_f[f_finding]:
            cand = by_f[f_finding].pop()
            if cand["subject_id"] not in used_f:
                f_anchor = cand
                break
        if f_anchor is None:
            stalls += 1
            continue
        # least-used retain finding that still has an unused subject
        r_candidates = list(by_r)
        rng.shuffle(r_candidates)
        r_candidates.sort(key=lambda f: retain_used[f])
        r_anchor = chosen_rf = None
        for rf in r_candidates:
            rejected = []
            while by_r[rf]:
                cand = by_r[rf].pop()
                if cand["subject_id"] in used_r:
                    continue
                if profiles_compatible(
                    f_anchor["subject_id"], cand["subject_id"],
                    subj_find, subj_fam, blocked_with,
                ):
                    r_anchor, chosen_rf = cand, rf
                    break
                rejected.append(cand)
            # Keep incompatible retain candidates for later forget anchors.
            by_r[rf].extend(rejected)
            if r_anchor is not None:
                break
        if r_anchor is None:
            stalls += 1
            continue
        stalls = 0
        used_f.add(f_anchor["subject_id"])
        used_r.add(r_anchor["subject_id"])
        forget_used[f_finding] += 1
        retain_used[chosen_rf] += 1
        pairs.append({
            "pair_id": f"EASY_{finding_abbr(f_anchor['target_finding'])}_"
                       f"{finding_abbr(r_anchor['target_finding'])}_{len(pairs)+1:04d}",
            "coupling_level": "easy_independent",
            "easy_mode": "concept_disjoint",
            "forget": anchor_view(f_anchor, sex_of),
            "retain": anchor_view(r_anchor, sex_of),
        })
    return pairs


def audit_easy_concept_disjoint(pairs, subj_find, subj_fam, blocked_with,
                                forget_findings, retain_findings,
                                report_by_study=None, support_device_re=None,
                                exclude_support_devices=True,
                                require_known_sex=True):
    """Global + patient-level concept-disjoint audit."""
    errs = []
    fset, rset = set(forget_findings), set(retain_findings)
    fsubj = {p["forget"]["subject_id"] for p in pairs}
    rsubj = {p["retain"]["subject_id"] for p in pairs}
    if fsubj & rsubj:
        errs.append(f"forget_retain_subject_overlap:{len(fsubj & rsubj)}")
    # global finding disjointness
    gf = {p["forget"]["target_finding"] for p in pairs}
    gr = {p["retain"]["target_finding"] for p in pairs}
    if gf & gr:
        errs.append(f"GLOBAL_finding_overlap:{sorted(gf & gr)}")
    if gf - fset:
        errs.append(f"forget_finding_off_set:{sorted(gf - fset)}")
    if gr - rset:
        errs.append(f"retain_finding_off_set:{sorted(gr - rset)}")
    # patient-level: each side's FULL profile stays on its side
    for p in pairs:
        a, b = p["forget"]["subject_id"], p["retain"]["subject_id"]
        if a == b:
            errs.append(f"{p['pair_id']}:same_subject")
        if subj_find[a] - fset:
            errs.append(f"{p['pair_id']}:forget_profile_leak:{sorted(subj_find[a]-fset)}")
        if subj_find[b] - rset:
            errs.append(f"{p['pair_id']}:retain_profile_leak:{sorted(subj_find[b]-rset)}")
        if not profiles_compatible(a, b, subj_find, subj_fam, blocked_with):
            errs.append(f"{p['pair_id']}:blocked_or_profile_overlap")
        if exclude_support_devices and support_device_re is not None and report_by_study is not None:
            f_text = report_text(report_by_study.get(p["forget"]["study_id"]))
            r_text = report_text(report_by_study.get(p["retain"]["study_id"]))
            if support_device_re.search(f_text):
                errs.append(f"{p['pair_id']}:forget_support_device_context")
            if support_device_re.search(r_text):
                errs.append(f"{p['pair_id']}:retain_support_device_context")
        if require_known_sex:
            if p["forget"].get("sex") is None:
                errs.append(f"{p['pair_id']}:forget_unknown_sex")
            if p["retain"].get("sex") is None:
                errs.append(f"{p['pair_id']}:retain_unknown_sex")
    return errs


# ============================================================
# HARD  (reads easy pool; reuse forget byte-identical, swap retain)
# ============================================================

def build_hard(easy_pairs, anchors, sex_of, n_pairs, seed,
               report_by_study=None, support_device_re=None,
               exclude_support_devices=True, require_known_sex=True):
    rng = random.Random(seed + 1)
    by_finding = defaultdict(list)
    for a in clean_anchors(anchors):           # clean anchors only (no `___`)
        if require_known_sex and sex_of.get(a["study_id"]) is None:
            continue
        if exclude_support_devices and support_device_re is not None and report_by_study is not None:
            if support_device_re.search(report_text(report_by_study.get(a["study_id"]))):
                continue
        by_finding[a["target_finding"]].append(a)
    for bucket in by_finding.values():
        rng.shuffle(bucket)

    # keep hard-retain identities disjoint from BOTH easy forget and easy retain
    excluded = {ep["forget"]["subject_id"] for ep in easy_pairs}
    excluded |= {ep["retain"]["subject_id"] for ep in easy_pairs}
    def match_score(f, c):
        # number of matching NON-null phenotype attrs (laterality/severity/location)
        s = 0
        for ax in ("laterality", "severity", "location"):
            fv, cv = f.get(ax), c.get(ax)
            if fv is not None and fv == cv:
                s += 1
        return s

    def is_l3(f, c):
        # same clinical phenotype: laterality AND severity both present & equal
        return (f.get("laterality") is not None and f.get("laterality") == c.get("laterality")
                and f.get("severity") is not None and f.get("severity") == c.get("severity"))

    pairs = []
    used_retain = set()
    for ep in easy_pairs:
        if len(pairs) >= n_pairs:
            break
        f = ep["forget"]  # reuse byte-identical
        finding = f["target_finding"]
        forget_subject = f["subject_id"]
        # candidates: same fine finding, different/unused/non-excluded subject
        cands = [c for c in by_finding.get(finding, [])
                 if c["subject_id"] != forget_subject
                 and c["subject_id"] not in used_retain
                 and c["subject_id"] not in excluded]
        if not cands:
            continue
        # pick the MOST phenotype-similar retain: l3 (laterality+severity match)
        # ranks first, then by number of matching attrs; rng-shuffled ties.
        chosen = max(cands, key=lambda c: (is_l3(f, c), match_score(f, c)))
        used_retain.add(chosen["subject_id"])
        level = "l3" if is_l3(f, chosen) else "l2"
        pairs.append({
            "pair_id": ep["pair_id"].replace("EASY_", "HARD_"),
            "coupling_level": level,
            "source_easy_pair_id": ep["pair_id"],
            "forget": f,  # byte-identical reuse
            "retain": anchor_view(chosen, sex_of),
        })
    return pairs


# ============================================================
# AUDIT + DIVERSITY
# ============================================================

def audit_easy(pairs, subj_find, subj_fam, blocked_with):
    """Patient-level strict audit: compare FULL subject profiles, not just the
    chosen anchors."""
    errs = []
    fsubj = {p["forget"]["subject_id"] for p in pairs}
    rsubj = {p["retain"]["subject_id"] for p in pairs}
    if fsubj & rsubj:
        errs.append(f"forget_retain_subject_overlap:{len(fsubj & rsubj)}")
    for p in pairs:
        a = p["forget"]["subject_id"]
        b = p["retain"]["subject_id"]
        if a == b:
            errs.append(f"{p['pair_id']}:same_subject")
        if subj_find[a] & subj_find[b]:
            errs.append(f"{p['pair_id']}:profile_fine_overlap:{sorted(subj_find[a] & subj_find[b])}")
        if subj_fam[a] & subj_fam[b]:
            errs.append(f"{p['pair_id']}:profile_family_overlap:{sorted(subj_fam[a] & subj_fam[b])}")
        if any(blocked_with.get(f, set()) & subj_fam[b] for f in subj_find[a]) or \
           any(blocked_with.get(f, set()) & subj_fam[a] for f in subj_find[b]):
            errs.append(f"{p['pair_id']}:blocked_edge")
    return errs


def audit_hard(pairs, report_by_study=None, support_device_re=None,
               exclude_support_devices=True, require_known_sex=True):
    errs = []
    fsubj = {p["forget"]["subject_id"] for p in pairs}
    rsubj = {p["retain"]["subject_id"] for p in pairs}
    if fsubj & rsubj:
        errs.append(f"forget_retain_subject_overlap:{len(fsubj & rsubj)}")
    for p in pairs:
        if p["forget"]["target_finding"] != p["retain"]["target_finding"]:
            errs.append(f"{p['pair_id']}:finding_mismatch")
        if p["forget"]["subject_id"] == p["retain"]["subject_id"]:
            errs.append(f"{p['pair_id']}:same_subject")
        if require_known_sex:
            if p["forget"].get("sex") is None:
                errs.append(f"{p['pair_id']}:forget_unknown_sex")
            if p["retain"].get("sex") is None:
                errs.append(f"{p['pair_id']}:retain_unknown_sex")
        if exclude_support_devices and support_device_re is not None and report_by_study is not None:
            f_text = report_text(report_by_study.get(p["forget"]["study_id"]))
            r_text = report_text(report_by_study.get(p["retain"]["study_id"]))
            if support_device_re.search(f_text):
                errs.append(f"{p['pair_id']}:forget_support_device_context")
            if support_device_re.search(r_text):
                errs.append(f"{p['pair_id']}:retain_support_device_context")
    return errs


def diversity(pairs):
    return {
        "count_pairs": len(pairs),
        "forget_finding": dict(Counter(p["forget"]["target_finding"] for p in pairs).most_common()),
        "retain_finding": dict(Counter(p["retain"]["target_finding"] for p in pairs).most_common()),
        "forget_family": dict(Counter(p["forget"]["target_family"] for p in pairs).most_common()),
        "retain_family": dict(Counter(p["retain"]["target_family"] for p in pairs).most_common()),
        "forget_sex": dict(Counter(p["forget"]["sex"] for p in pairs).most_common()),
        "retain_sex": dict(Counter(p["retain"]["sex"] for p in pairs).most_common()),
        "coupling_level": dict(Counter(p["coupling_level"] for p in pairs).most_common()),
        "pair_finding_combo": dict(Counter(
            f"{p['forget']['target_finding']} | {p['retain']['target_finding']}"
            for p in pairs).most_common(20)),
        # phenotype-match distribution (how coupled hard pairs actually are)
        "attr_match": {
            "same_laterality": sum(1 for p in pairs if p["forget"].get("laterality") is not None
                                   and p["forget"].get("laterality") == p["retain"].get("laterality")),
            "same_severity": sum(1 for p in pairs if p["forget"].get("severity") is not None
                                 and p["forget"].get("severity") == p["retain"].get("severity")),
            "same_location": sum(1 for p in pairs if p["forget"].get("location") is not None
                                 and p["forget"].get("location") == p["retain"].get("location")),
            "match_score_dist": dict(Counter(
                sum(1 for ax in ("laterality", "severity", "location")
                    if p["forget"].get(ax) is not None and p["forget"].get(ax) == p["retain"].get(ax))
                for p in pairs).most_common()),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Stage 20: build forget/retain pair pool")
    ap.add_argument("--coupling", choices=["easy", "hard"], default="easy")
    ap.add_argument("--anchors", default=DEFAULT_ANCHORS)
    ap.add_argument("--reports", default=DEFAULT_REPORTS)
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    ap.add_argument("--easy-pool", default=DEFAULT_EASY)
    ap.add_argument("--hard-pool", default=DEFAULT_HARD)
    ap.add_argument("--diversity", default=DEFAULT_DIVERSITY)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    # Global concept-disjoint easy: forget vs retain draw from disjoint finding
    # groups (so a retain-only `gold` model never sees the forget concepts).
    ap.add_argument("--concept-disjoint", action="store_true",
                    help="easy: globally disjoint forget/retain finding groups")
    ap.add_argument("--forget-findings", default=",".join(DEFAULT_FORGET_FINDINGS),
                    help="comma-separated forget findings (concept-disjoint mode)")
    ap.add_argument("--retain-findings", default=",".join(DEFAULT_RETAIN_FINDINGS),
                    help="comma-separated retain findings (concept-disjoint mode)")
    ap.add_argument("--allow-support-devices", action="store_true",
                    help="easy concept-disjoint: keep reports mentioning lines/tubes/devices")
    ap.add_argument("--allow-unknown-sex", action="store_true",
                    help="easy concept-disjoint: keep anchors whose sex is unknown after subject backfill")
    args = ap.parse_args()

    for path in (args.anchors, args.reports):
        if not os.path.exists(path):
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1
    family_of, blocked_with, rules, device_patterns = load_taxonomy(args.taxonomy)
    anchors = load_json(args.anchors)
    reports = load_json(args.reports)
    report_by_study = {r["study_id"]: r for r in reports}
    support_device_re = compile_phrase_re(device_patterns)
    sex_of = build_sex_map(reports, {a["study_id"] for a in anchors})
    subj_find, subj_fam = subject_profiles(anchors)   # patient-level profiles

    if args.coupling == "easy":
        if args.concept_disjoint:
            ff = [x.strip() for x in args.forget_findings.split(",") if x.strip()]
            rf = [x.strip() for x in args.retain_findings.split(",") if x.strip()]
            pairs = build_easy_concept_disjoint(
                anchors, subj_find, subj_fam, blocked_with, sex_of,
                args.n_pairs, args.seed, ff, rf, report_by_study, support_device_re,
                exclude_support_devices=not args.allow_support_devices,
                require_known_sex=not args.allow_unknown_sex,
            )
            errs = audit_easy_concept_disjoint(
                pairs, subj_find, subj_fam, blocked_with, ff, rf,
                report_by_study, support_device_re,
                exclude_support_devices=not args.allow_support_devices,
                require_known_sex=not args.allow_unknown_sex,
            )
        else:
            pairs = build_easy(anchors, subj_find, subj_fam, blocked_with, sex_of, args.n_pairs, args.seed)
            errs = audit_easy(pairs, subj_find, subj_fam, blocked_with)
        out = args.easy_pool
    else:
        if not os.path.exists(args.easy_pool):
            print(f"ERROR: --coupling hard needs {args.easy_pool} (build easy first)", file=sys.stderr)
            return 1
        easy_pairs = load_json(args.easy_pool)["pairs"]
        pairs = build_hard(
            easy_pairs, anchors, sex_of, args.n_pairs, args.seed,
            report_by_study, support_device_re,
            exclude_support_devices=not args.allow_support_devices,
            require_known_sex=not args.allow_unknown_sex,
        )
        errs = audit_hard(
            pairs, report_by_study, support_device_re,
            exclude_support_devices=not args.allow_support_devices,
            require_known_sex=not args.allow_unknown_sex,
        )
        out = args.hard_pool

    div = diversity(pairs)
    write_json(out, {
        "stage": "20_build_pair_pool",
        "coupling": args.coupling,
        "identity": "subject_id (real de-identified); sex from report; no fabricated names",
        "easy_overlap_rules": rules,
        "easy_context_filters": {
            "sex_source": "selected study report, else unambiguous same-subject report backfill",
            "exclude_support_devices": (
                (args.coupling == "hard" or args.concept_disjoint)
                and not args.allow_support_devices
            ),
            "require_known_sex": (
                (args.coupling == "hard" or args.concept_disjoint)
                and not args.allow_unknown_sex
            ),
        },
        "n_pairs": len(pairs),
        "audit_errors": errs,
        "diversity": div,
        "pairs": pairs,
    })
    div_all = load_json(args.diversity) if os.path.exists(args.diversity) else {}
    div_all[args.coupling] = div
    write_json(args.diversity, div_all)

    print(f"=== STAGE 20 [{args.coupling}] ===")
    print(f"requested {args.n_pairs} pairs, built {len(pairs)}")
    print(f"audit_errors: {len(errs)}" + ("" if not errs else f" -> {errs[:8]}"))
    print(f"forget_finding: {div['forget_finding']}")
    print(f"retain_finding: {div['retain_finding']}")
    print(f"forget_sex: {div['forget_sex']}  retain_sex: {div['retain_sex']}")
    if args.coupling == "hard":
        print(f"coupling_level: {div['coupling_level']}")
    print(f"wrote {out}")
    if len(pairs) < args.n_pairs:
        print(f"WARNING: only built {len(pairs)}/{args.n_pairs} pairs")
        return 2
    return 0 if not errs else 2


if __name__ == "__main__":
    raise SystemExit(main())
