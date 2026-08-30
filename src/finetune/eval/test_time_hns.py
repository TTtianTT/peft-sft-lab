"""Label-free task-level selection utilities for test-time HNS adapters.

The functions in this module deliberately operate on choice probabilities and
permutations only.  They do not accept labels, which keeps test-time selection
separate from the supervised accuracy computation used by the final evaluator.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class CandidateScore:
    """Unsupervised score for one candidate on one task."""

    entropy: float
    permutation_js: float
    reference_kl: float
    reference_flip_rate: float
    objective: float
    eligible: bool = True
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, float | bool | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionResult:
    selected_name: str
    reference_name: str
    reference_objective: float
    scores: dict[str, CandidateScore]


def make_choice_permutations(
    num_choices: int,
    num_permutations: int,
    *,
    seed: int,
) -> list[tuple[int, ...]]:
    """Return deterministic unique display->original choice permutations.

    The identity permutation is always first.  Requests larger than the number
    of possible permutations are capped, avoiding an unbounded retry loop.
    """
    if num_choices < 2:
        raise ValueError(f"num_choices must be >= 2, got {num_choices}")
    if num_permutations < 1:
        raise ValueError(f"num_permutations must be >= 1, got {num_permutations}")

    target = min(num_permutations, math.factorial(num_choices))
    identity = tuple(range(num_choices))
    permutations = [identity]
    seen = {identity}
    rng = random.Random(seed)
    while len(permutations) < target:
        candidate = list(identity)
        rng.shuffle(candidate)
        value = tuple(candidate)
        if value not in seen:
            seen.add(value)
            permutations.append(value)
    return permutations


def restore_original_choice_order(
    displayed_probabilities: torch.Tensor,
    display_to_original: Sequence[int],
) -> torch.Tensor:
    """Map probabilities over displayed letters back to original choice order."""
    if displayed_probabilities.ndim != 1:
        raise ValueError("displayed_probabilities must be one-dimensional")
    if displayed_probabilities.numel() != len(display_to_original):
        raise ValueError("probability/permutation lengths differ")
    if sorted(display_to_original) != list(range(len(display_to_original))):
        raise ValueError(f"Not a valid permutation: {tuple(display_to_original)}")

    restored = torch.empty_like(displayed_probabilities)
    for display_index, original_index in enumerate(display_to_original):
        restored[original_index] = displayed_probabilities[display_index]
    return restored


def _normalize_probabilities(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    probabilities = probabilities.detach().to(dtype=torch.float64).clamp_min(eps)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)


def _entropy(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)


def score_candidate_probabilities(
    probabilities: torch.Tensor,
    *,
    reference_probabilities: torch.Tensor,
    js_weight: float,
    reference_kl_weight: float,
    eps: float = 1e-12,
) -> CandidateScore:
    """Score ``[examples, permutations, choices]`` probabilities without labels."""
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [examples, permutations, choices]")
    if reference_probabilities.shape != probabilities.shape:
        raise ValueError("candidate and reference probability shapes differ")
    if probabilities.shape[-1] < 2:
        raise ValueError("at least two choices are required")

    probs = _normalize_probabilities(probabilities, eps)
    ref_probs = _normalize_probabilities(reference_probabilities, eps)
    mean_probs = probs.mean(dim=1)
    mean_probs = mean_probs / mean_probs.sum(dim=-1, keepdim=True)
    ref_mean = ref_probs.mean(dim=1)
    ref_mean = ref_mean / ref_mean.sum(dim=-1, keepdim=True)

    log_choices = math.log(probabilities.shape[-1])
    entropy = (_entropy(mean_probs, eps) / log_choices).mean()
    permutation_js = (
        (_entropy(mean_probs, eps) - _entropy(probs, eps).mean(dim=1)) / log_choices
    ).mean()
    reference_kl = (
        (
            mean_probs
            * (mean_probs.clamp_min(eps).log() - ref_mean.clamp_min(eps).log())
        ).sum(dim=-1)
        / log_choices
    ).mean()
    flip_rate = (mean_probs.argmax(dim=-1) != ref_mean.argmax(dim=-1)).to(torch.float64).mean()
    objective = entropy + js_weight * permutation_js + reference_kl_weight * reference_kl

    return CandidateScore(
        entropy=float(entropy.item()),
        permutation_js=float(permutation_js.item()),
        reference_kl=float(reference_kl.item()),
        reference_flip_rate=float(flip_rate.item()),
        objective=float(objective.item()),
    )


def score_candidate_probability_groups(
    probability_groups: Sequence[torch.Tensor],
    *,
    reference_probability_groups: Sequence[torch.Tensor],
    js_weight: float,
    reference_kl_weight: float,
) -> CandidateScore:
    """Score variable-choice-count groups using an example-weighted average."""
    if len(probability_groups) != len(reference_probability_groups):
        raise ValueError("candidate and reference group counts differ")
    if not probability_groups:
        raise ValueError("at least one probability group is required")

    weighted: list[tuple[int, CandidateScore]] = []
    for probabilities, reference in zip(probability_groups, reference_probability_groups):
        score = score_candidate_probabilities(
            probabilities,
            reference_probabilities=reference,
            js_weight=js_weight,
            reference_kl_weight=reference_kl_weight,
        )
        weighted.append((int(probabilities.shape[0]), score))

    total = sum(weight for weight, _ in weighted)
    if total <= 0:
        raise ValueError("probability groups contain zero examples")

    def average(field: str) -> float:
        return sum(weight * float(getattr(score, field)) for weight, score in weighted) / total

    return CandidateScore(
        entropy=average("entropy"),
        permutation_js=average("permutation_js"),
        reference_kl=average("reference_kl"),
        reference_flip_rate=average("reference_flip_rate"),
        objective=average("objective"),
    )


def select_candidate(
    candidate_probabilities: Mapping[str, torch.Tensor],
    *,
    reference_name: str,
    js_weight: float = 1.0,
    reference_kl_weight: float = 0.25,
    max_reference_kl: float = 0.10,
    min_improvement: float = 0.0,
) -> SelectionResult:
    """Select the lowest-risk candidate, falling back to the LoRA reference.

    A non-reference candidate must satisfy both the KL trust region and the
    requested objective improvement.  No ground-truth labels are accepted.
    """
    if reference_name not in candidate_probabilities:
        raise ValueError(f"Missing reference candidate {reference_name!r}")
    if max_reference_kl < 0:
        raise ValueError("max_reference_kl must be >= 0")
    if min_improvement < 0:
        raise ValueError("min_improvement must be >= 0")

    reference = candidate_probabilities[reference_name]
    raw_scores: dict[str, CandidateScore] = {}
    for name, probabilities in candidate_probabilities.items():
        raw_scores[name] = score_candidate_probabilities(
            probabilities,
            reference_probabilities=reference,
            js_weight=js_weight,
            reference_kl_weight=reference_kl_weight,
        )

    reference_objective = raw_scores[reference_name].objective
    scores: dict[str, CandidateScore] = {}
    selectable = [reference_name]
    for name, score in raw_scores.items():
        reason: str | None = None
        if name != reference_name and score.reference_kl > max_reference_kl:
            reason = "reference_kl_exceeds_limit"
        elif name != reference_name and score.objective > reference_objective - min_improvement:
            reason = "no_required_objective_improvement"
        eligible = reason is None
        scores[name] = CandidateScore(
            entropy=score.entropy,
            permutation_js=score.permutation_js,
            reference_kl=score.reference_kl,
            reference_flip_rate=score.reference_flip_rate,
            objective=score.objective,
            eligible=eligible,
            rejection_reason=reason,
        )
        if name != reference_name and eligible:
            selectable.append(name)

    selected_name = min(selectable, key=lambda name: (scores[name].objective, name))
    return SelectionResult(
        selected_name=selected_name,
        reference_name=reference_name,
        reference_objective=reference_objective,
        scores=scores,
    )


def select_candidate_grouped(
    candidate_probability_groups: Mapping[str, Sequence[torch.Tensor]],
    *,
    reference_name: str,
    js_weight: float = 1.0,
    reference_kl_weight: float = 0.25,
    max_reference_kl: float = 0.10,
    min_improvement: float = 0.0,
) -> SelectionResult:
    """Grouped counterpart of :func:`select_candidate` for variable choices."""
    if reference_name not in candidate_probability_groups:
        raise ValueError(f"Missing reference candidate {reference_name!r}")
    if max_reference_kl < 0:
        raise ValueError("max_reference_kl must be >= 0")
    if min_improvement < 0:
        raise ValueError("min_improvement must be >= 0")

    reference = candidate_probability_groups[reference_name]
    raw_scores = {
        name: score_candidate_probability_groups(
            groups,
            reference_probability_groups=reference,
            js_weight=js_weight,
            reference_kl_weight=reference_kl_weight,
        )
        for name, groups in candidate_probability_groups.items()
    }

    reference_objective = raw_scores[reference_name].objective
    scores: dict[str, CandidateScore] = {}
    selectable = [reference_name]
    for name, score in raw_scores.items():
        reason: str | None = None
        if name != reference_name and score.reference_kl > max_reference_kl:
            reason = "reference_kl_exceeds_limit"
        elif name != reference_name and score.objective > reference_objective - min_improvement:
            reason = "no_required_objective_improvement"
        eligible = reason is None
        scores[name] = CandidateScore(
            entropy=score.entropy,
            permutation_js=score.permutation_js,
            reference_kl=score.reference_kl,
            reference_flip_rate=score.reference_flip_rate,
            objective=score.objective,
            eligible=eligible,
            rejection_reason=reason,
        )
        if name != reference_name and eligible:
            selectable.append(name)

    selected_name = min(selectable, key=lambda name: (scores[name].objective, name))
    return SelectionResult(
        selected_name=selected_name,
        reference_name=reference_name,
        reference_objective=reference_objective,
        scores=scores,
    )
