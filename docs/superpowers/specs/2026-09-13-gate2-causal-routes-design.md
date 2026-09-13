# Gate 2 Design: Causal Transport Across Nonlinear Routes

Date: 2026-09-13

## Purpose

Gate 0 asked whether multiple hidden algorithms occupy reproducible residual geometry. Gate 1 asked whether the apparent inter-algorithm maps are richer than task-dependent translation. Gate 2 asks the causal question that those gates made meaningful:

> If a held-out affine map moves a residual state from one learned algorithm toward another, does that intervention cause the downstream nonlinear router to become target-like and the model's behavior to move toward the target algorithm?

Gate 2 also asks a narrower pruning question: are some MLP routes causally more important for one competent algorithm than the other, or is apparent route selectivity merely correlational?

The experiment remains CPU-only and must run on ordinary GitHub Actions. Scientific failure is valid and must be recorded faithfully.

## Frozen scientific context

Gate 2 preserves the Gate 1 task family, episode format, tokenizer, architecture, optimizer, seeds, loss definition, and training budget.

Model:

- 3 decoder blocks;
- width 48;
- 4 attention heads;
- MLP width 96;
- pre-LayerNorm;
- GELU MLP;
- causal softmax attention;
- residual connections;
- 3 demonstrations plus one query;
- tape length 4, modulus 8;
- query-output-only loss;
- 8000 AdamW training steps at learning rate 3e-4, batch size 32.

Training tasks remain COPY, REVERSE, SORT, CUMSUM_MOD, PREFIX_PARITY, and SWAP_PAIRS. DELTA_MOD remains held out.

Gate 1's completed run identified SORT and PREFIX_PARITY as the only tasks above the declared 0.80 exact-sequence competence threshold. Gate 2 therefore pre-registers one bidirectional pair:

- SORT -> PREFIX_PARITY
- PREFIX_PARITY -> SORT

The pair is fixed before Gate 2 is run. Gate 2 does not search over task pairs for the largest effect.

If either task falls below 0.80 exact accuracy in the Gate 2 retraining run, all causal metrics are still emitted but algorithmic interpretation is labeled ineligible.

## Route object

Gate 0/1 observe only residual states. Gate 2 exposes the state-dependent nonlinear routing variables that sit inside each decoder block.

For block `l`, let the pre-MLP normalized state be `u_l`, first MLP affine preactivation be

`z_l = u_l W1_l^T + b1_l`,

and GELU output be `g_l = GELU(z_l)`.

Gate 2 records at the final query separator:

1. `z_l` for all 96 hidden MLP units;
2. `GELU'(z_l)` for all 96 units, computed analytically;
3. per-head attention weights from the query separator to every visible prompt position.

The derivative vector is the primary MLP route fingerprint because the local MLP Jacobian contains the diagonal factor

`D_l(h) = diag(GELU'(z_l))`.

No claim is made that GELU creates literal discrete branches. Gate 2 treats these as smooth state-dependent route coefficients.

## Model instrumentation

`TinyTransformer.forward()` keeps its existing behavior by default. Gate 2 adds optional diagnostics and optional interventions without changing ordinary Gate 0/1 calls.

Diagnostics return, per block:

- MLP preactivation;
- GELU derivative;
- per-head attention weights.

Interventions are optional and scoped to one sequence position:

- residual patch after a selected block;
- optional MLP hidden-unit output mask for pruning tests.

With no intervention supplied, logits must be numerically identical to the existing model within floating-point tolerance.

## Paired banks

Gate 2 uses disjoint deterministic banks:

- affine fit bank: 192 paired query tapes;
- route reference bank: 192 paired query tapes;
- causal patch bank: 96 paired query tapes;
- pruning evaluation bank: 192 ordinary held-out episodes per focal task.

All banks are disjoint from training and from one another by fixed seed roots recorded in `config.json` and `RESULTS.md`.

The affine map is never fitted on the causal patch bank.

## Affine and translation maps

For each patch depth `l in {1, 2}` and each direction `(A, B)`, fit on the affine fit bank exactly as in Gate 1:

`H_B ~= H_A @ W_AB + b_AB`.

Also compute the translation null

`h_B_hat = h_A + (mu_B - mu_A)`.

The patch depths are residual layer 1 (after block 1) and residual layer 2 (after block 2). Layer 3 is excluded because no full transformer block remains downstream; layer 0 is excluded because Gate 1's task-conditioned effect is primarily a learned internal representation question.

## Causal patch experiment

For each source episode in the causal patch bank, first run the unmodified source prompt and capture its residual state at the selected patch layer and query-separator position.

Construct five interventions:

1. `identity`: no residual change;
2. `translation`: apply the Gate 1-style mean offset from source to target;
3. `affine`: apply the held-out fitted full affine source-to-target map;
4. `random_norm`: add a deterministic random direction whose displacement norm matches the affine displacement for that row;
5. `true_target`: replace the source query-separator residual with the actual target-prompt residual from the paired target episode at the same layer.

Only the query-separator residual at the selected depth is replaced. The source demonstrations and all other token states remain source-task context. This intentionally makes the intervention stringent: a successful target shift cannot be explained by swapping the demonstrations.

During autoregressive generation, the same query-separator patch is re-applied on every forward recomputation. Earlier causal states are otherwise unchanged.

## Behavioral metrics

For every direction, patch depth, and intervention report:

- exact sequence accuracy against the source task target;
- per-token accuracy against the source task target;
- exact sequence accuracy against the target task target;
- per-token accuracy against the target task target;
- target minus identity exact-accuracy shift;
- source minus identity exact-accuracy shift.

The strongest causal outcome would be an affine intervention that increases target-task behavior beyond both translation and norm-matched random controls while reducing or preserving source behavior in a coherent direction. A null or adverse result is valid.

`true_target` is an upper-bound / state-sufficiency control, not a learned-map baseline. If true-target transplantation does not shift behavior, failure of the affine patch is not evidence against the fitted map; it means the single-position residual is not sufficient under source context.

## Downstream route-shift metrics

For each patched run, collect route fingerprints in blocks downstream of the patch.

For every downstream block compare the patched route to the paired unmodified source and target routes.

For GELU derivative fingerprints report:

- cosine similarity to source route;
- cosine similarity to target route;
- target-minus-source cosine shift;
- normalized Euclidean distance to source and target.

For attention report the same comparisons on the flattened per-head query-separator attention vector, after zero-padding only when prompt lengths differ (normally they do not).

Primary route statistic:

`route_target_shift = similarity(patched, target) - similarity(patched, source)`.

An affine patch that produces a positive target shift beyond translation and random controls would support the claim that linear state transport crosses into a different downstream nonlinear routing regime.

## Route separability diagnostic

Before causal intervention, test whether route fingerprints themselves contain reproducible task information.

On the route reference bank, for each block:

- report mean within-task and between-task cosine distance for GELU derivative fingerprints;
- fit a closed-form ridge classifier on one deterministic half and test on the other half;
- repeat for flattened attention fingerprints;
- include shuffled-label controls.

This diagnostic is correlational and cannot establish causal routing by itself.

## Route-selective pruning

Gate 2 includes a deliberately small pruning intervention, not a general compression sweep.

For each block compute, on the route reference bank, each MLP unit's absolute difference in mean GELU derivative between SORT and PREFIX_PARITY:

`selectivity_i = abs(mean_SORT[d_i] - mean_PREFIX_PARITY[d_i])`.

At each block independently, mask the output of a fixed 10% of MLP hidden units (10 of 96, deterministic tie-breaking) under three strategies:

1. `most_selective`: units with highest selectivity;
2. `least_selective`: units with lowest selectivity;
3. `random`: deterministic random units with the same count.

Evaluate both focal tasks on the pruning bank without retraining. Report exact and token accuracy deltas from the unpruned model.

This is an exploratory causal check. Gate 2 does not claim that high derivative selectivity is an optimal pruning criterion. A useful result would be task-asymmetric damage from selective masking that exceeds least-selective and random controls.

## Required outputs

A Gate 2 run keeps the standard behavioral receipt and writes:

- `route_geometry.csv`
- `causal_transport.csv`
- `route_shift.csv`
- `pruning.csv`
- `gate2_route_shift.png`
- `gate2_causal_behavior.png`
- Gate 2 metric sections in `metrics.json`
- an explicit Gate 2 section in `RESULTS.md`

`schema_version` remains 1; Gate 2 extends the receipt without invalidating Gate 0/1 artifacts.

## Interpretation order

`RESULTS.md` must answer in this order:

1. Did SORT and PREFIX_PARITY both remain behaviorally competent?
2. Are GELU/attention route fingerprints reproducibly task-separable?
3. Can a true target residual transplant at layer 1 or 2 change behavior under source demonstrations?
4. Does the fitted affine patch change target behavior beyond translation and norm-matched random controls?
5. Does the affine patch move downstream GELU and/or attention routes toward the target route?
6. Does route-selective pruning cause task-asymmetric damage beyond controls?

The phrase "crosses a nonlinear route boundary" is eligible only if both behavior and downstream route fingerprints shift targetward beyond the declared controls. Because GELU and softmax are smooth, the receipt should prefer "routing regime" to "hard branch" unless a separate discontinuity experiment is added later.

## Engineering invariants

Unit tests must prove:

1. diagnostics-disabled forward output is unchanged;
2. a zero residual patch is equivalent to no patch when replacement equals the captured state;
3. an MLP all-ones mask is equivalent to no mask;
4. GELU derivative matches autograd on fixed test values;
5. causal patch banks are disjoint from affine fit banks;
6. random-norm controls match affine displacement norm within tolerance;
7. pruning strategies always mask exactly 10 of 96 units with deterministic indices;
8. Gate 0 and Gate 1 receipt validation remains backward compatible;
9. smoke CI stays cheap.

Scientific weakness never returns nonzero. NaNs, malformed artifacts, missing required files, or broken invariants remain engineering failures.

## GitHub Actions

Add a manual-only `.github/workflows/gate2.yml` with a 60-minute CPU timeout and artifact upload. The workflow runs:

`python -m transformer_study.experiment --preset gate2 --output artifacts/gate2`

then

`python -m transformer_study.experiment --validate artifacts/gate2`.

Normal push/PR CI runs unit tests and the existing cheap smoke experiment only. Gate 2 training must never run on ordinary pushes.

## Completion criteria

Gate 2 is complete when:

- all new and existing unit tests pass;
- ordinary CI is green;
- the manual Gate 2 workflow completes successfully;
- its artifact validates;
- `RESULTS.md` reports every causal and pruning control whether favorable or unfavorable;
- no interpretation is made when the focal task competence condition fails.
