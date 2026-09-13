# Gate 1 Design: Is Inter-Algorithm Geometry More Than Translation?

Date: 2026-09-13

## Purpose

Gate 0 found that held-out affine maps between task-conditioned residual states become much more accurate with depth and that map composition performs almost identically to a directly fitted map. However, the six task means are overwhelmingly low-dimensional and only one training task reached the declared competence threshold. Gate 1 asks the narrower question that Gate 0 made unavoidable:

> Are the apparent inter-algorithm maps doing anything materially richer than adding a task-dependent mean offset?

Gate 1 must remain CPU-only and run on ordinary GitHub Actions. A null result is a valid result.

## Frozen scientific constraints

Gate 1 does not change the task family, episode format, tokenizer, transformer architecture, seeds, loss definition, or residual extraction point.

The model remains:

- 3 decoder blocks;
- width 48;
- 4 attention heads;
- MLP width 96;
- 3 demonstrations plus one query;
- tape length 4, modulus 8;
- query-output-only loss;
- residual measured at the final query `<SEP>` before any query output token.

Training tasks remain COPY, REVERSE, SORT, CUMSUM_MOD, PREFIX_PARITY, and SWAP_PAIRS. DELTA_MOD remains held out.

The only training change is a fixed longer budget: `8000` steps with the same batch size 32 and optimizer settings. This is not an adaptive sweep. Gate 0 took only a few minutes on GitHub CPU, so the longer run remains comfortably inside the existing 60-minute workflow limit. The purpose is to test the same geometry in a regime where more than one task may be behaviorally competent.

Behavioral thresholds remain interpretation filters, never CI gates.

## Primary null model: translation only

For each ordered trained-task pair `(A, B)` and every residual layer, use the map-fit bank to compute task means

`mu_A = mean(H_A_fit)` and `mu_B = mean(H_B_fit)`.

The translation-only predictor on held-out states is

`h_B_hat = h_A + (mu_B - mu_A)`.

This is the critical Gate 1 null because task-offset geometry automatically composes:

`(mu_B - mu_A) + (mu_C - mu_B) = mu_C - mu_A`.

Therefore Gate 0's strong composition result is not evidence of an operator algebra unless a richer map beats this null on held-out states.

## Map hierarchy

For every layer and ordered pair, report the same held-out test rows for the following map family.

1. **Identity**: `h_B_hat = h_A`.
2. **Translation**: `h_B_hat = h_A + mu_B - mu_A`.
3. **Low-rank centered correction**, ranks `1, 2, 4, 8`.
4. **Full affine ridge**, identical to Gate 0.
5. Existing mean-predictor and random-pair controls remain available for context.

The low-rank correction is defined without allowing rank truncation to hide a mean shift. Center the fit states:

`Xc = H_A_fit - mu_A`

`Yc = H_B_fit - mu_B`.

Fit the centered ridge map `Yc ~= Xc @ M`. Let `C = M - I`. Compute the SVD of `C` and keep the top `k` singular components, `C_k`. The rank-k predictor is

`h_B_hat = mu_B + (h_A - mu_A) @ (I + C_k)`.

Rank therefore measures transformation complexity *beyond translation*.

## Primary metrics

For each pair and layer report:

- identity NMSE;
- translation NMSE;
- low-rank NMSE for ranks 1, 2, 4, 8;
- full-affine NMSE;
- cosine similarity for translation and full affine;
- absolute gain `translation_nmse - affine_nmse`;
- relative gain `(translation_nmse - affine_nmse) / translation_nmse` with numerical protection;
- the smallest correction rank that recovers at least 90% of the full-affine improvement over translation, when a positive improvement exists.

The primary Gate 1 scientific statistic is the held-out full-affine gain over translation, not the full-affine score by itself.

## Centered-state diagnostic

Translation-only prediction is equivalent to assuming that, after removing task means, the source and target states share the same geometry.

Therefore explicitly report

`centered_identity_nmse = NMSE(H_A_test - mu_A, H_B_test - mu_B)`.

Also fit and report the full centered linear map. If translation performs nearly as well as full affine and centered identity is already strong, the Gate 0 result is primarily a task-offset phenomenon.

If full affine materially improves over translation, especially with low-rank corrections that generalize, then task-conditioned computation changes the occupied residual geometry rather than merely moving a shared input representation.

## Correct-behavior conditioning

Gate 0 was mostly a mixed-competence experiment. Gate 1 must separate geometry of correct computation from geometry of failure.

The maps are always fit on the complete fit bank. On the disjoint test bank, additionally evaluate each pair on rows where **both the source task and target task were answered exactly correctly**.

Emit the correct-conditioned result only when at least `20` paired test rows satisfy that condition; otherwise mark the row `underpowered` and record the sample count. Do not fit a separate 48-dimensional map on a tiny correct-only subset.

This keeps selection out of training while asking whether the same map survives on examples where both computations actually succeeded.

## Composition control

For every ordered triple `(A, B, C)`, Gate 1 reports three held-out action errors:

1. direct full affine `A -> C`;
2. composed full affine `(A -> B) -> (B -> C)`;
3. translation composition.

Translation composition is evaluated explicitly even though its offsets telescope algebraically. Its action error equals the direct `A -> C` translation error up to floating-point precision and serves as the exact null explanation for compositionality.

Also report composition after removing means using the centered fitted matrices:

`M_AB @ M_BC` versus `M_AC`.

The meaningful composition question is therefore whether **centered transformations** compose on held-out occupied states better than a translation-only model can explain.

## Competence interpretation

Gate 1 uses the fixed 8000-step training budget and reports the same exact/token accuracies as Gate 0.

Interpretation levels:

- If fewer than two trained tasks reach exact accuracy `>= 0.80`, Gate 1 can characterize representation geometry but cannot support a claim about relations among multiple learned algorithms.
- If at least two trained tasks are competent, pairwise correct-conditioned rows between competent tasks become eligible for algorithmic interpretation.
- If three or more are competent, centered composition among competent triples becomes eligible for the strongest Gate 1 interpretation.

These conditions affect prose only. They never make the workflow fail.

DELTA_MOD novelty remains reported for continuity but is not the primary Gate 1 question.

## Outputs

The Gate 1 run keeps every Gate 0 artifact and adds:

- `translation_controls.csv` — pair/layer identity, translation, low-rank, full-affine comparison;
- `correct_conditioned.csv` — pair/layer metrics restricted to both-correct test rows, or an underpowered marker;
- `composition_controls.csv` — full-affine, centered, and translation composition controls;
- a Gate 1 section in `metrics.json` under `translation_controls`, `correct_conditioned`, and `composition_controls`;
- a compact plot showing translation NMSE versus full-affine NMSE by layer and a plot of relative affine gain over translation.

`RESULTS.md` must answer, in order:

1. How many trained tasks are behaviorally competent?
2. Does translation explain most of the inter-task map accuracy?
3. How much extra held-out gain does full affine provide?
4. What rank beyond translation is sufficient when there is a gain?
5. Does the gain persist on both-correct examples?
6. Does centered composition add evidence beyond the exactly compositional translation null?

## Receipt and compatibility

Keep `schema_version = 1`; Gate 1 extends the receipt with additional keys rather than redefining existing Gate 0 fields.

`validate_receipt()` reads `config.json`. For `preset == "gate1"`, it additionally requires the three Gate 1 CSV files and the three Gate 1 metric keys. Gate 0 and smoke receipts remain valid without those files.

Scientific weakness must never cause a nonzero process exit. NaNs, malformed outputs, missing files, or broken invariants remain engineering failures.

## GitHub Actions

Add `gate1` to the experiment CLI presets and add a manual-only `.github/workflows/gate1.yml` with the same 60-minute CPU limit and artifact upload behavior as Gate 0.

Normal push/PR CI runs only unit tests and the existing tiny smoke preset. It must not run 8000-step training.

## Gate 1 completion criteria

Gate 1 is complete when:

1. unit tests prove translation offsets telescope exactly;
2. synthetic tests prove low-rank centered corrections recover known rank-k transformations;
3. tests prove full affine and translation metrics use only the disjoint test bank;
4. correct-conditioned rows are emitted only at `n >= 20`;
5. smoke CI remains cheap and green;
6. the full Gate 1 GitHub CPU workflow produces a valid receipt;
7. `RESULTS.md` explicitly states whether the Gate 0 linear/compositional effect survives the translation null.

The desired hypothesis is not a completion criterion.

## Out of scope

Gate 1 does not causally inject transformed residual states back into the transformer. That is the natural Gate 2 only if Gate 1 finds geometry beyond translation. It also does not change model architecture, add task tokens, tune the task suite, run multiple seeds, or search hyperparameters.