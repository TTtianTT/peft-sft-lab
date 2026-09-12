# Why can HNS exceed the Base model?

## Protocol

- Greedy Base -> LoRA -> original-HNS transitions were paired by item for every off-task edge in the completed 2-base x 4-task forgetting matrix.
- The stochastic audit preselected two symmetric edges per base: Magicoder adapter -> GSM8K and MetaMath adapter -> HumanEval.
- Each item received 8 rollouts at temperature 0.7 and top-p 0.95 under Base, original LoRA/HNS, common-basis LoRA/HNS, and a target-feasible calibrated global scalar.
- GSM8K used all 1,319 test items; HumanEval used all 164 items. vLLM batch invariance was enabled and all variants shared the base tokenizer and prompt implementation.
- All reported intervals below are paired item-bootstrap 95% intervals with 20,000 draws.

## Greedy item-transition identity

For any edge,

`HNS - Base = preserved positive transfer + HNS novel gain - persistent forgetting - HNS new harm`.

Thus, recovery of LoRA forgetting explains HNS minus LoRA, but cannot by itself make HNS exceed Base. Above-Base performance requires retained LoRA transfer and/or genuinely new HNS successes to outweigh residual forgetting and new damage.

| Base | Train -> eval | Base | LoRA | HNS | Positive side: preserved transfer + novel | Negative side: persistent forgetting + new harm | Net HNS-Base |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | Magicoder -> GSM8K | 85.67 | 84.15 | 88.86 | 54 + 24 = 78 | 18 + 18 = 36 | +42 / 1,319 |
| Qwen3-8B | MetaMath -> HumanEval | 66.46 | 64.02 | 67.68 | 14 + 3 = 17 | 14 + 1 = 15 | +2 / 164 |
| Llama-3.1-8B | Magicoder -> GSM8K | 62.40 | 56.41 | 62.47 | 44 + 48 = 92 | 55 + 36 = 91 | +1 / 1,319 |
| Llama-3.1-8B | MetaMath -> HumanEval | 52.44 | 54.27 | 60.37 | 19 + 6 = 25 | 12 + 0 = 12 | +13 / 164 |

Across all 24 off-task edges (including item-weighted Commonsense subtasks), HNS recovered 4,951 LoRA-forgotten outcomes, preserved 3,590 positive-transfer outcomes, and created 898 Base-wrong/LoRA-wrong/HNS-correct outcomes. These raw totals are dominated by the much larger Commonsense suite; edge-level and subtask-level tables must be used for comparisons.

## Stochastic audit

Expected reward is the mean correctness of the 8 rollouts per item.

| Base | Train -> eval | Base | Original LoRA | Original HNS | Common LoRA | Common HNS | Calibrated scalar |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | Magicoder -> GSM8K | 86.15 | 80.75 | 88.43 | 80.56 | **88.67** | 86.94 (gamma=0.40) |
| Qwen3-8B | MetaMath -> HumanEval | 66.62 | 59.76 | 67.15 | 59.22 | **67.76** | 65.85 (gamma=0.40) |
| Llama-3.1-8B | Magicoder -> GSM8K | **63.53** | 48.84 | 61.64 | 49.73 | 62.35 | 61.45 (gamma=0.25) |
| Llama-3.1-8B | MetaMath -> HumanEval | 44.74 | 51.45 | 52.82 | 51.60 | **53.05** | 52.44 (gamma=0.40) |

Key paired comparisons for common-basis HNS:

| Base | Train -> eval | HNS - Base | HNS - LoRA | HNS - scalar |
|---|---|---:|---:|---:|
| Qwen3-8B | Magicoder -> GSM8K | **+2.51** [1.75, 3.29] | **+8.10** [7.01, 9.20] | **+1.72** [0.96, 2.51] |
| Qwen3-8B | MetaMath -> HumanEval | +1.14 [-4.57, 6.78] | **+8.54** [3.96, 13.26] | +1.91 [-0.69, 4.73] |
| Llama-3.1-8B | Magicoder -> GSM8K | **-1.18** [-2.22, -0.17] | **+12.62** [11.37, 13.86] | **+0.90** [0.06, 1.74] |
| Llama-3.1-8B | MetaMath -> HumanEval | **+8.31** [4.34, 12.50] | +1.45 [-0.69, 3.58] | +0.61 [-1.14, 2.36] |

## Interpretation

1. **Qwen Magicoder -> GSM8K is the clean spectral-balancing positive case.** HNS exceeds Base and a calibrated scalar in expected reward. Invalid-answer rates are negligible (Base 0.23%, common HNS 0.03%, scalar 0.00%), so the gain is not primarily formatting. On the 24 greedy `HNS novel gain` items, stochastic correctness rises from 45.8% for Base and 39.1% for LoRA to 59.9% for HNS. This supports a genuine redistribution of probability toward correct solutions.

2. **Llama MetaMath -> HumanEval exceeds Base mostly because useful LoRA transfer survives.** Common HNS is +8.31 pp over Base, but only +1.45 pp over LoRA and +0.61 pp over scalar, with both latter intervals crossing zero. HNS appears to retain mathematical/programmatic transfer while reducing some harmful amplitude; the present experiment does not establish a shape-specific advantage there.

3. **Llama Magicoder -> GSM8K exposes decoding dependence.** Greedy HNS was essentially tied with Base. Under stochastic sampling, common HNS has lower mean reward than Base (-1.18 pp), yet higher majority-vote accuracy (+2.43 pp, CI [0.53, 4.32]) and higher any@8 (+2.27 pp, CI [0.61, 3.94]). HNS redistributes per-question success probabilities rather than uniformly improving them. Claims that it “beats Base” must name the decoding/aggregation protocol.

4. **Qwen MetaMath -> HumanEval remains underpowered for HNS versus Base.** The point estimate is positive and HNS strongly repairs LoRA, but 164 problems leave a wide HNS-Base interval. It supports restoration, not a confident above-Base claim.

5. **Factor representation is not the main explanation in three of four edges.** Original and common-basis HNS expected rewards differ by -0.24, -0.61, and -0.23 pp with intervals crossing zero for both Qwen edges and Llama HumanEval. Llama GSM8K shows a small -0.71 pp original-minus-common difference, so common-basis comparisons remain the primary mechanism estimate.

## Conclusion

HNS can exceed Base through a two-part effect: it removes harmful parts of a task adapter while retaining transferable task information. In the strongest Qwen GSM8K case, it also creates a robust shape-specific improvement beyond calibrated scalar shrinkage, consistent with spectral reallocation moving probability toward correct reasoning. This is not universal: another apparent above-Base case changes sign under stochastic expected reward, and the HumanEval cases cannot separate HNS from scalar at current power.

The defensible claim is therefore conditional: **HNS sometimes produces a better transfer/forgetting trade-off than both Base and LoRA, but whether this appears as above-Base performance depends on the checkpoint, task, scalar control, and decoding functional.**
