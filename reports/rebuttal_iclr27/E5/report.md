# E5: LoDA scope assessment

Public source is available in [the authors' repository](https://github.com/HHHLF/LoDA_ICML2026/tree/e34d169189933a63600f37aa78b21864c08a7b59). Its [network implementation](https://github.com/HHHLF/LoDA_ICML2026/blob/e34d169189933a63600f37aa78b21864c08a7b59/models/network.py) uses a pretrained Vision Transformer; published configs cover image classification sequences. Porting it to Qwen causal-language-model training would require a new implementation and recipe. Under the requested setting-mismatch exception, no GPU comparison is run.

[LoDA](https://proceedings.mlr.press/v306/he26f.html) uses task data and past feature statistics to construct shared/isolated subspaces, learns up-projections, and recalibrates before merging task updates. LoRA-Norm edits a completed single adapter without data. These interventions address different stages; combining them is a future hypothesis, not an established compatibility or additive-gain result.

Rebuttal wording: “LoDA is a data-dependent continual-learning training method; our experiment studies data-free post-processing of a fixed single-task adapter. We clarify this scope difference and do not infer a performance ranking across these settings.”

No metrics or CIs exist for this assessment; both TSVs deliberately contain headers only.
