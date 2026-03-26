# Learned Routers Don't Learn: MoE Miscalibration

**Paper:** [Learned Routers Don't Learn: Expert Miscalibration in MoE Models](https://github.com/Lumi-node/model-garage/raw/main/research/papers/moe-miscalibration/Learned_Routers_Dont_Learn_MoE_Miscalibration_Young2026.pdf)

## Summary

Per-layer expert isolation reveals that learned MoE routers show **Spearman rho = 0.069** between routing probability and expert quality. Despite **482/896** expert-layer-domain combinations showing statistically significant specialization (BH FDR, alpha=0.05), the router largely ignores this specialization.

## Key Finding

MoE models have experts that genuinely specialize — but the learned router doesn't route tokens to the best expert for the task. The correlation between "how often the router picks an expert" and "how good that expert actually is" is near zero.

```
Router says: "Expert 2 is best for this token"
Reality:      Expert 5 performs best for this domain
Correlation:  rho = 0.069 (essentially random)
```

## Methodology

Using Model Garage's `analyze` module, the paper:

1. **Isolated individual experts** by hooking into MoE layers and routing all tokens to a single expert
2. **Evaluated each expert** on domain-specific benchmarks
3. **Compared expert quality rankings** against the router's routing probabilities
4. **Applied statistical testing** (Benjamini-Hochberg FDR correction, alpha=0.05)

## Key Results

- **482/896** expert-layer-domain combinations show statistically significant specialization
- Router probability vs expert quality correlation: **rho = 0.069**
- Expert 2 (E2) shows dominance bias — receives disproportionate routing regardless of task
- v4.0 experiment with 5x more evaluation data, per-layer confidence intervals, and random baselines confirms findings

## Implications

1. **Current MoE routers are suboptimal** — there's significant room for improvement
2. **Expert specialization is real** — the capacity for domain-specific routing exists
3. **Router redesign could yield large gains** — quality-aware routing could substantially improve MoE performance
4. **Model Garage enables this analysis** — per-expert isolation at scale was previously impractical

## Model Garage Modules Used

- `analyze` — Activation analysis and expert performance measurement
- `core.hooks` — Layer-level interception for expert isolation
- `registry` — Model decomposition and expert enumeration
