# Model Selection & Architecture Notes

This document records the available Monet checkpoints, the architectural
details that matter for the conversion pipeline, and the resulting decisions
about which models to target at each stage.

## Available Checkpoints

The authors (Park, Ahn, Kim, Kang; DMIS Lab, Korea University) released nine
checkpoints on HuggingFace under the `MonetLLM/` namespace. All checkpoints
are trained on 100B tokens (indicated by `100BT` in the name).

### Base language models

| Hub ID                          | Decomposition | Params |
| ------------------------------- | ------------- | ------ |
| `MonetLLM/monet-vd-850M-100BT-hf` | vertical      | 850M   |
| `MonetLLM/monet-hd-850M-100BT-hf` | horizontal    | 850M   |
| `MonetLLM/monet-vd-1.4B-100BT-hf` | vertical      | 1.4B   |
| `MonetLLM/monet-hd-1.4B-100BT-hf` | horizontal    | 1.4B   |
| `MonetLLM/monet-vd-4.1B-100BT-hf` | vertical      | 4.1B   |
| `MonetLLM/monet-hd-4.1B-100BT-hf` | horizontal    | 4.1B   |

### Specialized variants (all VD)

| Hub ID                                 | Notes                             |
| -------------------------------------- | --------------------------------- |
| `MonetLLM/monet-vd-1.4B-100BT-chat-hf` | Instruction-tuned chat version    |
| `MonetLLM/codemonet-vd-1.4B-100BT-hf`  | Code-specialized variant          |
| `MonetLLM/visionmonet-vd-1.4B-100BT-hf`| Vision-language variant (~2B tot) |

## Architecture Details That Matter for Conversion

All base models share the same core design:

- **262,144 effective experts per layer**, realized via a product-key
  factorization. Each layer maintains `N = 512` **half-experts**, and the
  effective expert is formed by the Cartesian product of two half-expert
  selections, yielding `N^2 = 262_144` distinct expert functions.
- Total parameters scale as `O(N)` in the number of half-experts, not
  `O(N^2)` in the number of effective experts. This is the whole point of
  Monet's decomposition.

Per-scale configuration:

| Scale | `d_model` | `d_expert` |
| ----- | --------- | ---------- |
| 850M  | 1536      | 12         |
| 1.4B  | 2048      | 16         |
| 4.1B  | 3072      | 24         |

The individual expert is a **12–24 dimensional subspace**. This is tiny
compared to a normal FFN hidden and is very good news for exact tree
extraction (Aytekin): the decision tree size scales with the number of
effective neurons, and these experts have effectively 12–24 "neurons" worth
of nonlinearity to capture.

### VD vs HD

**VD (vertical decomposition)** partitions each expert's input/output
dimensions into left and right segments. The two halves of an effective
expert operate on disjoint dimension slices, then compose additively.

**HD (horizontal decomposition)** partitions the experts themselves into
bottom and top halves. The router picks one bottom-half and one top-half
and they chain: the output distribution of the top-half depends on which
bottom-half was selected upstream.

Implications for conversion:

- **VD half-experts are independent functions** of disjoint inputs. Each
  half-expert can be converted in isolation and the effective-expert output
  is a simple sum. Input dimension per half-expert is smaller than the full
  residual, which makes exact tree extraction even cheaper.
- **HD half-experts are dynamically composed.** Converting HD experts means
  converting both halves **and** handling the fact that the top-half's
  effective input distribution depends on which bottom-half was selected.
  This is a meaningfully more entangled problem — the converted top-half
  circuit would need to condition on (or be specialized per) the
  bottom-half selection.

The paper reports VD consistently outperforming HD on quality, and every
specialized variant (chat, code, vision) is VD. **We target VD throughout**
and treat HD only as a late-stage comparison point.

## Why the "262k experts per layer" headline number is misleading for us

The headline count is `N^2` effective experts, but because the decomposition
is a product key there are only `2N = 1024` **half-experts** per layer to
convert. At 6–12 transformer layers per model that gives on the order of
10k total circuits to build per model, not `262_144 × num_layers` (≈ 1.6M+).

This is tractable enough that per-half-expert distillation is viable even
without amortizing conversion via a learned converter model. The learned
converter (Step 3b) remains valuable for:

1. Finding shorter circuits than direct distillation would.
2. Sharing structure across half-experts with similar input/output behavior.
3. Amortizing minimization cost across the population.

But the upper-bound "how many circuits do we need to build?" number is
~10k, not ~1.6M. The math previously assumed was off by roughly 2 orders
of magnitude in the wrong direction.

## Half-expert vs effective-expert conversion

Two conversion granularities are possible:

1. **Convert each of the `N^2` effective experts independently.** Simple
   but wasteful — most of these circuits share structure because they're
   built from the same pool of half-experts. Building ~262k circuits per
   layer is also just larger than necessary.
2. **Convert the `2N` half-experts and the composition rule separately.**
   Preserves Monet's parameter-efficiency advantage in the converted form.
   Circuit count scales as `O(N) = O(sqrt(effective experts))`, matching
   Monet's original parameter scaling.

Option 2 is the target for a mature implementation. Option 1 is a useful
intermediate target for development: it is conceptually simple and lets us
measure per-effective-expert conversion quality without worrying about
composition correctness.

**Plan: start development on option 2 directly for VD** because VD
half-experts are genuinely independent (disjoint input slices composed
additively), which makes the half-expert approach no harder than
per-effective-expert for VD. The code already needs to distinguish the
two axes of the product-key router, so we may as well build the wrapper
layer around half-experts from the start.

## Recommended Model Progression

### Primary development target: `MonetLLM/monet-vd-850M-100BT-hf`

Smallest VD checkpoint. Everything runs fastest, iterations are cheapest,
and `d_expert = 12` is maximal smallness — the tightest case for Aytekin
to succeed. Use for Steps 0, 1, 2, and initial Step 3a/3b development.

### Scaling target: `MonetLLM/monet-vd-1.4B-100BT-hf`

Step up once 850M pipeline is stable. `d_expert = 16` is still small and
this is the size that most of the paper's analyses use, so numbers can be
compared directly. Only scale with chat and code variants, which matters
for demonstrating the method transfers to specialized models.

### Full-scale validation: `MonetLLM/monet-vd-4.1B-100BT-hf`

Run only once the pipeline is stable. `d_expert = 24` is where exact tree
extraction may start to strain. If conversion works here we have a
genuinely interesting result; if it breaks here but worked at 1.4B, the
failure mode tells us something about `d_expert` scaling of the approach.

### Demonstration targets (downstream of base models working)

- `MonetLLM/monet-vd-1.4B-100BT-chat-hf` — "does this preserve
  instruction-following?"
- `MonetLLM/codemonet-vd-1.4B-100BT-hf` — "does this preserve specialized
  capabilities?"
- `MonetLLM/visionmonet-vd-1.4B-100BT-hf` — deferred; vision adds
  orthogonal interface issues.

### Deferred / comparison-only

- All `monet-hd-*` checkpoints. Only revisited as a late-stage "does this
  generalize beyond VD?" comparison point.

## Go/No-Go by model scale

- **850M pipeline broken:** abort — something is wrong with the approach,
  not with scaling.
- **850M works, 1.4B broken:** investigate `d_expert` sensitivity; possibly
  revise quantization or tree-extraction heuristics before scaling further.
- **1.4B works, 4.1B broken:** publishable negative result on `d_expert`
  scaling; investigate whether selective fallback (Step 3d) recovers it.
- **All three scales work:** move to specialized variants as the
  publishable demonstration story.
