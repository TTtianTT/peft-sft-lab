#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFEval evaluation (prompt/inst × strict/loose), driven by instruction_id_list + kwargs.

This script supports **Scheme 1** integration with the official Google IFEval evaluator:
- It keeps your current generation logic (Transformers or vLLM),
- It writes official-format JSONL files:
    1) input_data.jsonl (key/prompt/instruction_id_list/kwargs)
    2) input_response_data.jsonl (prompt/response)
- Then it runs:
    python -m instruction_following_eval.evaluation_main \
      --input_data=... --input_response_data=... --output_dir=...

Notes:
- Your own strict/loose checker + outputs.jsonl + metrics.json are preserved.
- Official outputs will be written under: {output_dir}/official_ifeval/

Expected official code layout:
  <OFFICIAL_EVAL_ROOT>/
    instruction_following_eval/
      evaluation_main.py
      instructions.py
      ...

You can point --official_eval_root either to:
  - <OFFICIAL_EVAL_ROOT> (parent of instruction_following_eval/), or
  - <OFFICIAL_EVAL_ROOT>/instruction_following_eval (the package directory itself).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from finetune.eval.generation import (
    generate_greedy,
    generate_greedy_vllm_batch,
    load_transformers_model,
    save_json,
    strip_code_fences,
)
from finetune.utils import seed_everything


# ----------------------------
# Optional deps
# ----------------------------
HAVE_LANGDETECT = False
try:
    from langdetect import detect  # type: ignore

    HAVE_LANGDETECT = True
except Exception:
    detect = None


# ----------------------------
# Dataset
# ----------------------------
def _load_ifeval_dataset(split: str):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc
    # Official HF id is google/IFEval (capital IFEval)
    return load_dataset("google/IFEval", split=split)


# ----------------------------
# Text utils
# ----------------------------
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+", flags=re.UNICODE)
_ALLCAP_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
_TITLE_RE = re.compile(r"<<[^<>]+>>")
_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")

_DIVIDER_LINE_RE = re.compile(r"^\s*\*\s*\*\s*\*\s*$", flags=re.MULTILINE)
_BULLET_LINE_RE = re.compile(r"^\s*\*\s+\S+", flags=re.MULTILINE)

# Try to count italic/bold segments. This is heuristic, but matches the benchmark's intent.
_HIGHLIGHT_RE = re.compile(
    r"(\*\*[^*\n][^*\n]*\*\*)|(\*[^*\n][^*\n]*\*)",
    flags=re.UNICODE,
)


def _norm_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _strip_outer_ws(s: str) -> str:
    return s.strip(" \t\n")


def _count_words(s: str) -> int:
    return len(_WORD_RE.findall(s))


def _count_sentences(s: str) -> int:
    parts = _SENT_SPLIT_RE.split(s.strip())
    return len([p for p in parts if p.strip()])


def _split_paragraphs_by_divider(s: str) -> List[str]:
    s = _norm_newlines(s)
    parts = _DIVIDER_LINE_RE.split(s)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs_by_blanklines(s: str) -> List[str]:
    s = _norm_newlines(s).strip()
    # "two line breaks" → split on blank lines
    parts = re.split(r"\n\s*\n+", s)
    return [p.strip() for p in parts if p.strip()]


def _first_word(s: str) -> str:
    s = s.strip()
    m = re.search(r"\b(\w+)\b", s)
    return m.group(1) if m else ""


def _count_allcaps_words(s: str) -> int:
    return len(_ALLCAP_WORD_RE.findall(s))


def _count_placeholders(s: str) -> int:
    return len(_PLACEHOLDER_RE.findall(s))


def _count_bullets(s: str) -> int:
    # Count "* " bullet lines, but exclude divider "* * *"
    bullets = 0
    for line in _norm_newlines(s).split("\n"):
        if _DIVIDER_LINE_RE.match(line):
            continue
        if re.match(r"^\s*\*\s+\S+", line):
            bullets += 1
    return bullets


def _count_highlights(s: str) -> int:
    return len([m for m in _HIGHLIGHT_RE.finditer(s)])


def _lang_code_from_name(name: str) -> Optional[str]:
    # Dataset sometimes uses "English" etc.
    n = name.strip().lower()
    mapping = {
        "english": "en",
        "en": "en",
        "spanish": "es",
        "es": "es",
        "french": "fr",
        "fr": "fr",
        "german": "de",
        "de": "de",
        "italian": "it",
        "it": "it",
        "portuguese": "pt",
        "pt": "pt",
        "japanese": "ja",
        "ja": "ja",
        "korean": "ko",
        "ko": "ko",
        "chinese": "zh-cn",
        "zh": "zh-cn",
        "russian": "ru",
        "ru": "ru",
        "arabic": "ar",
        "ar": "ar",
    }
    return mapping.get(n)


def _detect_language_code(text: str) -> Optional[str]:
    if not HAVE_LANGDETECT:
        return None
    try:
        code = detect(text)
        return str(code).lower()
    except Exception:
        return None


# ----------------------------
# Loose transformations (8 variants)
# ----------------------------
def _t_remove_markdown_asterisks(s: str) -> str:
    # Paper explicitly mentions removing commonly seen font modifiers, especially * and **.
    return s.replace("**", "").replace("*", "")


def _t_remove_first_line(s: str) -> str:
    lines = _norm_newlines(s).split("\n")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[1:])


def _t_remove_last_line(s: str) -> str:
    lines = _norm_newlines(s).split("\n")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[:-1])


def _loose_variants(s: str) -> List[str]:
    # Identity + 3 transforms + pairwise + all three = 8
    v0 = s
    v1 = _t_remove_markdown_asterisks(s)
    v2 = _t_remove_first_line(s)
    v3 = _t_remove_last_line(s)

    def c(a: Callable[[str], str], b: Callable[[str], str]) -> str:
        return b(a(s))

    def c3(a: Callable[[str], str], b: Callable[[str], str], c_: Callable[[str], str]) -> str:
        return c_(b(a(s)))

    return [
        v0,
        v1,
        v2,
        v3,
        c(_t_remove_markdown_asterisks, _t_remove_first_line),
        c(_t_remove_markdown_asterisks, _t_remove_last_line),
        c(_t_remove_first_line, _t_remove_last_line),
        c3(_t_remove_markdown_asterisks, _t_remove_first_line, _t_remove_last_line),
    ]


# ----------------------------
# Instruction checking
# ----------------------------
def _norm_inst_id(inst_id: str) -> Tuple[str, str]:
    s = inst_id.strip().lower().replace("-", "_")
    if ":" in s:
        a, b = s.split(":", 1)
        return a.strip(), b.strip()
    return "unknown", s


def _get_relation(kwargs: Dict[str, Any]) -> Optional[str]:
    rel = kwargs.get("relation")
    if isinstance(rel, str) and rel.strip():
        return rel.strip().lower()
    return None


def _compare_count(count: int, target: int, relation: Optional[str]) -> bool:
    # relation strings in dataset examples include: "at least", "less than"
    if relation is None:
        return count == target
    r = relation.strip().lower()
    if r in {"at least", ">=", "no less than"}:
        return count >= target
    if r in {"at most", "<=", "no more than"}:
        return count <= target
    if r in {"less than", "<"}:
        return count < target
    if r in {"more than", ">"}:
        return count > target
    if r in {"exactly", "==", "="}:
        return count == target
    if r in {"around", "approximately", "about"}:
        tol = max(1, int(round(0.10 * target)))
        return abs(count - target) <= tol
    # Fallback: default to equality (fail-closed-ish but stable)
    return count == target


@dataclass
class Check:
    passed: bool
    detail: Dict[str, Any]


def _check_keywords_existence(resp: str, kwargs: Dict[str, Any]) -> Check:
    kws = kwargs.get("keywords")
    if kws is None:
        kw = kwargs.get("keyword")
        kws = [kw] if kw else []
    if not isinstance(kws, list):
        kws = [kws]
    missing = []
    text = resp
    for k in kws:
        if k is None:
            continue
        k = str(k)
        if not k.strip():
            continue
        # whole-word if simple token, substring if phrase
        if re.fullmatch(r"[A-Za-z0-9_]+", k):
            pat = re.compile(rf"\b{re.escape(k)}\b", flags=re.IGNORECASE)
            if not pat.search(text):
                missing.append(k)
        else:
            if k.lower() not in text.lower():
                missing.append(k)
    return Check(passed=(len(missing) == 0), detail={"missing": missing, "keywords": kws})


def _check_keyword_frequency(resp: str, kwargs: Dict[str, Any]) -> Check:
    kw = kwargs.get("keyword")
    n = kwargs.get("frequency")
    rel = _get_relation(kwargs)
    if kw is None or n is None:
        return Check(False, {"error": "missing keyword/frequency"})
    kw = str(kw)
    n = int(n)
    if re.fullmatch(r"[A-Za-z0-9_]+", kw):
        pat = re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE)
        c = len(pat.findall(resp))
    else:
        c = resp.lower().count(kw.lower())
    return Check(_compare_count(c, n, rel), {"keyword": kw, "count": c, "target": n, "relation": rel})


def _check_forbidden_words(resp: str, kwargs: Dict[str, Any]) -> Check:
    bad = kwargs.get("forbidden_words") or []
    if not isinstance(bad, list):
        bad = [bad]
    found = []
    for w in bad:
        if w is None:
            continue
        w = str(w)
        if not w.strip():
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", w):
            pat = re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
            if pat.search(resp):
                found.append(w)
        else:
            if w.lower() in resp.lower():
                found.append(w)
    return Check(passed=(len(found) == 0), detail={"found": found, "forbidden_words": bad})


def _check_letter_frequency(resp: str, kwargs: Dict[str, Any]) -> Check:
    letter = kwargs.get("letter")
    n = kwargs.get("let_frequency")
    rel = kwargs.get("let_relation")
    if letter is None or n is None:
        return Check(False, {"error": "missing letter/let_frequency"})
    letter = str(letter)
    if len(letter) != 1:
        # If dataset gives something unexpected, fail closed.
        return Check(False, {"error": "letter is not a single char", "letter": letter})
    n = int(n)
    c = _norm_newlines(resp).lower().count(letter.lower())
    rel_s = str(rel).lower() if isinstance(rel, str) and rel.strip() else None
    return Check(_compare_count(c, n, rel_s), {"letter": letter, "count": c, "target": n, "relation": rel_s})


def _check_response_language(resp: str, kwargs: Dict[str, Any]) -> Check:
    lang = kwargs.get("language")
    if not isinstance(lang, str) or not lang.strip():
        return Check(False, {"error": "missing language"})
    want = _lang_code_from_name(lang) or lang.strip().lower()
    got = _detect_language_code(resp)
    if got is None:
        # If langdetect isn't installed, do not silently pass.
        return Check(False, {"error": "langdetect not installed", "wanted": want})
    passed = (got == want) or (want.startswith("zh") and got.startswith("zh"))
    return Check(passed, {"wanted": want, "detected": got})


def _check_number_words(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("num_words")
    rel = _get_relation(kwargs)
    if n is None:
        return Check(False, {"error": "missing num_words"})
    n = int(n)
    c = _count_words(resp)
    return Check(_compare_count(c, n, rel), {"count": c, "target": n, "relation": rel})


def _check_number_sentences(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("num_sentences")
    rel = _get_relation(kwargs)
    if n is None:
        return Check(False, {"error": "missing num_sentences"})
    n = int(n)
    c = _count_sentences(resp)
    return Check(_compare_count(c, n, rel), {"count": c, "target": n, "relation": rel})


def _check_number_paragraphs(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("num_paragraphs")
    if n is None:
        return Check(False, {"error": "missing num_paragraphs"})
    n = int(n)
    paras = _split_paragraphs_by_divider(resp)
    if not paras:
        paras = _split_paragraphs_by_blanklines(resp)
    c = len(paras)
    # number_paragraphs instruction is "should contain N paragraphs" → exact
    return Check(c == n, {"count": c, "target": n})


def _check_nth_paragraph_first_word(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("num_paragraphs")
    i = kwargs.get("nth_paragraph")
    first = kwargs.get("first_word")
    if n is None or i is None or first is None:
        return Check(False, {"error": "missing num_paragraphs/nth_paragraph/first_word"})
    n = int(n)
    i = int(i)
    first = str(first)
    paras = _split_paragraphs_by_blanklines(resp)
    ok_n = (len(paras) == n)
    if not ok_n or i < 1 or i > len(paras):
        return Check(False, {"count": len(paras), "target_paragraphs": n, "nth": i, "first_word": first})
    got = _first_word(paras[i - 1])
    ok_first = (got.lower() == first.lower())
    return Check(
        ok_n and ok_first,
        {"count": len(paras), "target_paragraphs": n, "nth": i, "expected_first": first, "got_first": got},
    )


def _check_postscript(resp: str, kwargs: Dict[str, Any]) -> Check:
    marker = kwargs.get("postscript_marker")
    if not isinstance(marker, str) or not marker.strip():
        return Check(False, {"error": "missing postscript_marker"})
    marker = marker.strip()
    lines = [ln.rstrip() for ln in _norm_newlines(resp).split("\n")]
    # last non-empty line should start with marker
    last_nonempty = ""
    for ln in reversed(lines):
        if ln.strip():
            last_nonempty = ln.strip()
            break
    passed = last_nonempty.startswith(marker)
    return Check(passed, {"marker": marker, "last_nonempty_line": last_nonempty})


def _check_number_placeholders(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("num_placeholders")
    if n is None:
        return Check(False, {"error": "missing num_placeholders"})
    n = int(n)
    c = _count_placeholders(resp)
    return Check(c >= n, {"count": c, "min_required": n})


def _check_number_bullets(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("num_bullets")
    if n is None:
        return Check(False, {"error": "missing num_bullets"})
    n = int(n)
    c = _count_bullets(resp)
    return Check(c == n, {"count": c, "target": n})


def _check_title(resp: str, kwargs: Dict[str, Any]) -> Check:
    m = _TITLE_RE.search(resp)
    return Check(bool(m), {"matched": m.group(0) if m else None})


def _check_choose_from(resp: str, kwargs: Dict[str, Any]) -> Check:
    # Dataset doesn't expose an explicit "options" field; commonly stored in `keywords`
    opts = kwargs.get("keywords")
    if opts is None:
        kw = kwargs.get("keyword")
        opts = [kw] if kw else []
    if not isinstance(opts, list):
        opts = [opts]
    opts = [str(o) for o in opts if o is not None and str(o).strip()]
    out = _strip_outer_ws(resp)
    passed = any(out == o for o in opts)
    return Check(passed, {"output": out, "options": opts})


def _check_number_highlighted_sections(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("num_highlights")
    if n is None:
        return Check(False, {"error": "missing num_highlights"})
    n = int(n)
    c = _count_highlights(resp)
    return Check(c >= n, {"count": c, "min_required": n})


def _check_multiple_sections(resp: str, kwargs: Dict[str, Any]) -> Check:
    spl = kwargs.get("section_spliter")  # dataset uses this key (typo in schema)
    n = kwargs.get("num_sections")
    if spl is None or n is None:
        return Check(False, {"error": "missing section_spliter/num_sections"})
    spl = str(spl).strip()
    n = int(n)
    # Count lines that look like "{spl} X" where X is an integer
    pat = re.compile(rf"^\s*{re.escape(spl)}\s+(\d+)\b", flags=re.IGNORECASE | re.MULTILINE)
    hits = [int(m.group(1)) for m in pat.finditer(resp)]
    passed = (len(hits) == n)
    return Check(passed, {"splitter": spl, "count": len(hits), "target": n, "hits": hits[:10]})


def _check_json_format(resp: str, kwargs: Dict[str, Any]) -> Check:
    s = _strip_outer_ws(strip_code_fences(resp))
    try:
        obj = json.loads(s)
        return Check(True, {"parsed_type": type(obj).__name__})
    except Exception as e:
        return Check(False, {"error": str(e)})


def _check_repeat_prompt(resp: str, prompt: str, kwargs: Dict[str, Any]) -> Check:
    to_repeat = kwargs.get("prompt_to_repeat")
    if not isinstance(to_repeat, str) or not to_repeat.strip():
        return Check(False, {"error": "missing prompt_to_repeat"})
    out = _norm_newlines(resp).lstrip()
    rep = _norm_newlines(to_repeat).strip()
    passed = out.startswith(rep)
    tail = out[len(rep) :] if passed else ""
    # Must have some answer after repeating (non-empty after stripping)
    has_answer = bool(tail.strip())
    return Check(passed and has_answer, {"starts_with_repeat": passed, "has_answer_after": has_answer})


def _check_two_responses(resp: str, kwargs: Dict[str, Any]) -> Check:
    delim = "******"
    parts = resp.split(delim)
    if len(parts) != 2:
        return Check(False, {"error": "delimiter not found exactly once", "num_parts": len(parts)})
    a, b = parts[0].strip(), parts[1].strip()
    passed = bool(a) and bool(b) and (a != b)
    return Check(passed, {"a_len": len(a), "b_len": len(b), "different": a != b})


def _check_english_capital(resp: str, kwargs: Dict[str, Any]) -> Check:
    # No lowercase letters allowed
    has_lower = bool(re.search(r"[a-z]", resp))
    passed = not has_lower
    return Check(passed, {"has_lowercase": has_lower})


def _check_english_lowercase(resp: str, kwargs: Dict[str, Any]) -> Check:
    # No capital letters allowed
    has_upper = bool(re.search(r"[A-Z]", resp))
    passed = not has_upper
    return Check(passed, {"has_uppercase": has_upper})


def _check_capital_word_frequency(resp: str, kwargs: Dict[str, Any]) -> Check:
    n = kwargs.get("capital_frequency")
    rel = kwargs.get("capital_relation")
    if n is None or rel is None:
        return Check(False, {"error": "missing capital_frequency/capital_relation"})
    n = int(n)
    rel_s = str(rel).strip().lower()
    c = _count_allcaps_words(resp)
    return Check(_compare_count(c, n, rel_s), {"count": c, "target": n, "relation": rel_s})


def _check_end_checker(resp: str, kwargs: Dict[str, Any]) -> Check:
    phrase = kwargs.get("end_phrase")
    if not isinstance(phrase, str) or not phrase.strip():
        return Check(False, {"error": "missing end_phrase"})
    phrase = phrase.strip()
    out = _norm_newlines(resp).rstrip()
    passed = out.endswith(phrase)
    return Check(passed, {"end_phrase": phrase})


def _check_quotation(resp: str, kwargs: Dict[str, Any]) -> Check:
    out = _strip_outer_ws(resp)
    passed = (len(out) >= 2 and out[0] == '"' and out[-1] == '"')
    return Check(passed, {"first_char": out[:1], "last_char": out[-1:] if out else ""})


def _check_no_comma(resp: str, kwargs: Dict[str, Any]) -> Check:
    passed = ("," not in resp)
    return Check(passed, {"has_comma": ("," in resp)})


def check_instruction_strict(
    *,
    inst_id: str,
    resp: str,
    prompt: str,
    kwargs: Dict[str, Any],
) -> Check:
    cat, name = _norm_inst_id(inst_id)

    # keywords
    if cat == "keywords" and name == "existence":
        return _check_keywords_existence(resp, kwargs)
    # HF dataset uses keywords:frequency
    if cat == "keywords" and name in {"frequency", "keyword_frequency"}:
        return _check_keyword_frequency(resp, kwargs)
    if cat == "keywords" and name == "forbidden_words":
        return _check_forbidden_words(resp, kwargs)
    if cat == "keywords" and name == "letter_frequency":
        return _check_letter_frequency(resp, kwargs)

    # language
    if cat == "language" and name == "response_language":
        return _check_response_language(resp, kwargs)

    # length constraints
    if cat in {"length_constraints", "length_constraint"} and name == "number_words":
        return _check_number_words(resp, kwargs)
    if cat in {"length_constraints", "length_constraint"} and name == "number_sentences":
        return _check_number_sentences(resp, kwargs)
    if cat in {"length_constraints", "length_constraint"} and name == "number_paragraphs":
        # Some dataset variants may still use number_paragraphs with nth/first_word in kwargs
        if kwargs.get("nth_paragraph") is not None:
            return _check_nth_paragraph_first_word(resp, kwargs)
        return _check_number_paragraphs(resp, kwargs)
    if cat in {"length_constraints", "length_constraint"} and name in {"nth_paragraph_first_word", "paragraph_first_word"}:
        return _check_nth_paragraph_first_word(resp, kwargs)

    # detectable content
    if cat == "detectable_content" and name == "postscript":
        return _check_postscript(resp, kwargs)
    if cat == "detectable_content" and name in {"number_placeholders", "number_placeholder"}:
        return _check_number_placeholders(resp, kwargs)

    # detectable format
    if cat == "detectable_format" and name in {"number_bullet_lists", "number_bullets", "number_bullet_points"}:
        return _check_number_bullets(resp, kwargs)
    if cat == "detectable_format" and name == "title":
        return _check_title(resp, kwargs)
    if cat == "detectable_format" and name == "choose_from":
        return _check_choose_from(resp, kwargs)
    if cat == "detectable_format" and name == "number_highlighted_sections":
        return _check_number_highlighted_sections(resp, kwargs)
    if cat == "detectable_format" and name == "multiple_sections":
        return _check_multiple_sections(resp, kwargs)
    if cat == "detectable_format" and name in {"json_format", "json"}:
        return _check_json_format(resp, kwargs)

    # combination
    if cat == "combination" and name == "repeat_prompt":
        return _check_repeat_prompt(resp, prompt, kwargs)
    if cat == "combination" and name == "two_responses":
        return _check_two_responses(resp, kwargs)

    # change case
    if cat in {"change_case", "change_cases"} and name == "english_capital":
        return _check_english_capital(resp, kwargs)
    if cat in {"change_case", "change_cases"} and name == "english_lowercase":
        return _check_english_lowercase(resp, kwargs)
    if cat in {"change_case", "change_cases"} and name == "capital_word_frequency":
        return _check_capital_word_frequency(resp, kwargs)

    # start/end
    if cat in {"startend", "start_end"} and name == "end_checker":
        return _check_end_checker(resp, kwargs)
    if cat in {"startend", "start_end"} and name == "quotation":
        return _check_quotation(resp, kwargs)

    # punctuation
    if cat == "punctuation" and name in {"no_comma", "no_commas"}:
        return _check_no_comma(resp, kwargs)

    return Check(False, {"error": f"unknown instruction id: {inst_id}", "cat": cat, "name": name})


def check_instruction_loose(
    *,
    inst_id: str,
    resp: str,
    prompt: str,
    kwargs: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    # Any of the 8 transformed variants can satisfy the strict checker
    variants = _loose_variants(resp)
    for idx, v in enumerate(variants):
        ck = check_instruction_strict(inst_id=inst_id, resp=v, prompt=prompt, kwargs=kwargs)
        if ck.passed:
            return True, {"passed_variant": idx, "detail": ck.detail}
    # Keep the strict detail on the untransformed text for debugging
    ck0 = check_instruction_strict(inst_id=inst_id, resp=resp, prompt=prompt, kwargs=kwargs)
    return False, {"passed_variant": None, "detail": ck0.detail}


# ----------------------------
# Official eval (Scheme 1)
# ----------------------------
def _dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _resolve_official_pkg_parent(official_eval_root: str) -> Path:
    """
    Returns a directory that should be added to PYTHONPATH so that
    `instruction_following_eval` becomes importable.
    """
    root = Path(official_eval_root).expanduser().resolve()
    if (root / "instruction_following_eval").is_dir():
        return root
    if root.is_dir() and root.name == "instruction_following_eval":
        return root.parent
    raise RuntimeError(
        f"Invalid --official_eval_root={official_eval_root}. "
        f"Expected either a directory containing instruction_following_eval/, "
        f"or the instruction_following_eval/ directory itself."
    )


def run_official_ifeval(
    *,
    out_dir: Path,
    records: List[Dict[str, Any]],
    generations: List[str],
    official_eval_root: str,
    official_input_data: Optional[str] = None,
    extra_official_flags: Optional[List[str]] = None,
) -> Path:
    """
    Exports official-format JSONLs and runs the official evaluator via subprocess.

    Returns: path to official output dir.
    """
    official_out = out_dir / "official_ifeval"
    official_out.mkdir(parents=True, exist_ok=True)

    input_data_path = official_out / "input_data.jsonl"
    input_resp_path = official_out / "input_response_data.jsonl"

    # 1) input_data.jsonl (key/prompt/instruction_id_list/kwargs)
    data_rows: List[Dict[str, Any]] = []
    for r in records:
        kw_clean: List[Dict[str, Any]] = []
        for kw in r["kwargs"]:
            kwd = dict(kw) if isinstance(kw, dict) else dict(kw)
            # remove None keys to be safe
            for k in list(kwd.keys()):
                if kwd[k] is None:
                    kwd.pop(k, None)
            kw_clean.append(kwd)

        data_rows.append(
            {
                "key": r.get("key"),
                "prompt": r["prompt"],
                "instruction_id_list": list(r["instruction_id_list"]),
                "kwargs": kw_clean,
            }
        )
    _dump_jsonl(input_data_path, data_rows)

    # 2) input_response_data.jsonl (prompt/response)
    # To be robust, we include both "response" and "output" (harmless extra key).
    resp_rows = [
        {"prompt": r["prompt"], "response": g, "output": g, "key": r.get("key")}
        for r, g in zip(records, generations)
    ]
    _dump_jsonl(input_resp_path, resp_rows)

    # 3) Run official evaluator
    pkg_parent = _resolve_official_pkg_parent(official_eval_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(pkg_parent) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        f"--input_data={official_input_data or str(input_data_path)}",
        f"--input_response_data={str(input_resp_path)}",
        f"--output_dir={str(official_out)}",
    ]
    if extra_official_flags:
        cmd.extend(extra_official_flags)

    # Some absl-based scripts behave better when run with cwd=pkg_parent
    subprocess.run(cmd, check=True, env=env, cwd=str(pkg_parent))
    return official_out


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate IFEval (prompt/inst × strict/loose).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)

    p.add_argument("--per_category_metrics", action="store_true")
    p.add_argument("--fail_on_unknown", action="store_true", help="Raise if an unknown instruction id appears.")

    # Scheme 1: official evaluation
    p.add_argument("--run_official_eval", action="store_true", help="Run official IFEval evaluator after generation.")
    p.add_argument(
        "--official_eval_root",
        type=str,
        default=None,
        help="Path to directory containing instruction_following_eval/ (or the package dir itself).",
    )
    p.add_argument(
        "--official_input_data",
        type=str,
        default=None,
        help="Optional: use an existing official input_data.jsonl instead of exporting from HF dataset.",
    )
    p.add_argument(
        "--official_extra_flags",
        type=str,
        nargs="*",
        default=None,
        help="Optional: extra flags forwarded to official evaluation_main (e.g., --use_cached_results=...).",
    )
    return p


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "outputs.jsonl"

    ds = _load_ifeval_dataset(args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    loaded = None
    if not args.use_vllm:
        loaded = load_transformers_model(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            dtype=args.dtype,
            device_map="auto",
        )

    # Aggregates
    n_prompts = 0
    n_prompt_strict_ok = 0
    n_prompt_loose_ok = 0

    n_insts = 0
    n_inst_strict_ok = 0
    n_inst_loose_ok = 0

    per_cat = defaultdict(lambda: Counter())  # cat -> Counter(strict_ok, loose_ok, total)

    # Prepare generations
    prompts: List[str] = []
    records: List[Dict[str, Any]] = []

    for ex in ds:
        prompt = str(ex["prompt"])
        inst_ids = list(ex.get("instruction_id_list") or [])
        kwargs_list = list(ex.get("kwargs") or [])
        if len(inst_ids) != len(kwargs_list):
            raise RuntimeError(
                f"Mismatch: {len(inst_ids)} instruction_ids vs {len(kwargs_list)} kwargs for key={ex.get('key')}"
            )
        prompts.append(prompt)
        records.append({"key": ex.get("key"), "prompt": prompt, "instruction_id_list": inst_ids, "kwargs": kwargs_list})

    if args.use_vllm:
        generations = generate_greedy_vllm_batch(
            base_model=args.base_model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            adapter_dir=args.adapter_dir,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    else:
        generations = []
        for r in records:
            gen = generate_greedy(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                prompt=r["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
            generations.append(gen)

    # Your existing JSONL outputs + your own metrics
    with outputs_path.open("w", encoding="utf-8") as f:
        for r, gen in zip(records, generations):
            prompt = r["prompt"]
            inst_ids: List[str] = r["instruction_id_list"]
            kwargs_list: List[Dict[str, Any]] = r["kwargs"]

            inst_results = []
            prompt_strict_ok = True
            prompt_loose_ok = True

            for inst_id, kw in zip(inst_ids, kwargs_list):
                if not isinstance(kw, dict):
                    kw = dict(kw)

                if args.fail_on_unknown:
                    ck0 = check_instruction_strict(inst_id=inst_id, resp=gen, prompt=prompt, kwargs=kw)
                    if (not ck0.passed) and isinstance(ck0.detail, dict) and "unknown instruction id" in str(
                        ck0.detail.get("error", "")
                    ):
                        raise RuntimeError(f"Unknown instruction id encountered: {inst_id}")

                ck_strict = check_instruction_strict(inst_id=inst_id, resp=gen, prompt=prompt, kwargs=kw)
                ck_loose_passed, ck_loose_meta = check_instruction_loose(inst_id=inst_id, resp=gen, prompt=prompt, kwargs=kw)

                inst_results.append(
                    {
                        "instruction_id": inst_id,
                        "kwargs": kw,
                        "strict_passed": bool(ck_strict.passed),
                        "strict_detail": ck_strict.detail,
                        "loose_passed": bool(ck_loose_passed),
                        "loose_detail": ck_loose_meta,
                    }
                )

                n_insts += 1
                n_inst_strict_ok += int(ck_strict.passed)
                n_inst_loose_ok += int(ck_loose_passed)

                cat, _ = _norm_inst_id(inst_id)
                per_cat[cat]["total"] += 1
                per_cat[cat]["strict_ok"] += int(ck_strict.passed)
                per_cat[cat]["loose_ok"] += int(ck_loose_passed)

                prompt_strict_ok = prompt_strict_ok and bool(ck_strict.passed)
                prompt_loose_ok = prompt_loose_ok and bool(ck_loose_passed)

            n_prompts += 1
            n_prompt_strict_ok += int(prompt_strict_ok)
            n_prompt_loose_ok += int(prompt_loose_ok)

            f.write(
                json.dumps(
                    {
                        "key": r["key"],
                        "prompt": prompt,
                        "output": gen,
                        "prompt_strict_passed": prompt_strict_ok,
                        "prompt_loose_passed": prompt_loose_ok,
                        "inst_results": inst_results,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics: Dict[str, Any] = {
        "prompt_level_strict_acc": (n_prompt_strict_ok / n_prompts if n_prompts else 0.0),
        "inst_level_strict_acc": (n_inst_strict_ok / n_insts if n_insts else 0.0),
        "prompt_level_loose_acc": (n_prompt_loose_ok / n_prompts if n_prompts else 0.0),
        "inst_level_loose_acc": (n_inst_loose_ok / n_insts if n_insts else 0.0),
        "total_prompts": n_prompts,
        "total_instructions": n_insts,
        "avg_instructions_per_prompt": (n_insts / n_prompts if n_prompts else 0.0),
        "use_vllm": bool(args.use_vllm),
        "langdetect_available": bool(HAVE_LANGDETECT),
    }

    if args.per_category_metrics:
        cat_metrics = {}
        for cat, ctr in per_cat.items():
            total = int(ctr["total"])
            cat_metrics[cat] = {
                "total": total,
                "strict_acc": (ctr["strict_ok"] / total if total else 0.0),
                "loose_acc": (ctr["loose_ok"] / total if total else 0.0),
            }
        metrics["per_category"] = cat_metrics

    save_json(out_dir / "metrics.json", metrics)

    # Scheme 1: run official evaluator (subprocess)
    if args.run_official_eval:
        if not args.official_eval_root:
            raise RuntimeError("--official_eval_root is required when --run_official_eval is set.")
        run_official_ifeval(
            out_dir=out_dir,
            records=records,
            generations=generations,
            official_eval_root=args.official_eval_root,
            official_input_data=args.official_input_data,
            extra_official_flags=args.official_extra_flags,
        )


if __name__ == "__main__":
    main()
