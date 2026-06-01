# Biomechanical Safety — Complete Implementation Plan

> **Audience**: engineering team + reviewing AI assistant + consulting physiotherapist.
> **Scope**: from the current prototype (`app/safety/`) to a 100%-safe, production-rolled-out
> AI workout generator with the legacy Three-Tier + Excluder pipeline retired.
> **Reading order**: §1 (executive summary) → §3 (senior-review fixes) → §4 (web-search containment, the high-risk gap) → §6 (phased roadmap).

---

## 1. Executive Summary

| Metric | Current Three-Tier | Target Vetted Manifold |
|---|---|---|
| Per-request LLM calls (safety) | 3 (excluder, planner, safety review) | 1 (intake translator) |
| Per-request tokens (safety overhead) | ~5,500 | ~1,700 |
| Per-request latency | 6–8 s | ~3 s |
| Safety leak surface | string keyword matching at 4 layers | zero — structured-enum filter only |
| Web-search-injected exercises | unbounded, can reach user | physically cannot reach structured workout output |
| Refusal capability | none (fallbacks always fire) | first-class, segment-coverage-based |
| Library tagging cost | $0 | ~2 days for full 2,800-exercise library |
| Compute cost in steady state | higher | lower |

The investment is one-time data engineering (tagging schema) and one-time architectural
work. After that, the safety floor is structural, not probabilistic.

---

## 2. Prototype Status (What's Done)

In `app/safety/`:

- ✅ `schema.py` — Pydantic enums for the 7 features + InjuryConstraint with the
  `blocked_chains` modification. Compiles, validated.
- ✅ `filter.py` — pure-Python deterministic filter with audit trail
  (`safety_violations()`), fast path (`is_safe()`), and evaluation hook
  (`filter_with_audit()`).
- ✅ `intake.py` — LLM injury → constraint translator with fail-safe fallback.
- ✅ `tags_lower_body.json` — 41 hand-tagged exercises across 8 movement families.
- ✅ Validation: all 5 injury scenarios produce expected behavior. PFPS chain-blocking
  works as designed.

---

## 3. Critical Fixes from Senior Review

The senior architect's review identified **three structural tagging leaks**, **two missed
biomechanical risks**, and **one architectural refinement**. All are required before v1.0.

### 3.1 Tagging leaks (apply to `tags_lower_body.json`)

| Exercise ID | Current Tag | Bug | Fix |
|---|---|---|---|
| `plank`, `side_plank` | `primary_joints_involved: []` | Heavy isometric LUMBAR_SPINE bracing. A user with acute L4-L5 disc herniation (`blocked_joints: [LUMBAR_SPINE]`) gets prescribed plank → pelvic sag → shear force on injured disc. | Add `LUMBAR_SPINE` to `primary_joints_involved`. |
| `lying_leg_raise` | `primary_joints_involved: [HIP]` | Notorious for pelvic anterior tilt and lumbar hyperextension under fatigue. Common PT contraindication for low-back patients. | Add `LUMBAR_SPINE` to `primary_joints_involved`. |
| `hip_thrust` | `grip_requirement: NONE` | Heavy barbell sits on hips but requires active hand stabilization to prevent tilt/roll. Severe wrist injury would be unsafe. | Upgrade to `grip_requirement: LIGHT`. |

**Note on `plank`**: the alternative — adding an explicit `trunk_bracing_required: bool`
field — was considered. The simpler fix (tag LUMBAR_SPINE as a prime mover) achieves the
same safety outcome without a schema change. Trade-off: planks are no longer prescribed for
users with ANY lumbar injury, even chronic-managed cases where light isometric bracing is
actually rehab. Acceptable conservative bias for v1.

### 3.2 Missed biomechanical risks (add to `schema.py`)

#### 3.2.1 Torsional/Rotational Joint Loading

**Risk**: `reverse_lunge` and `curtsy_lunge` currently carry **identical** tags despite
radically different injury profiles. Curtsy lunge forces the knee into the frontal and
transverse planes (valgus + rotational stress). An ACL-recovering user is safe with linear
reverse lunges but at high re-injury risk with curtsy lunges.

**Schema addition**:
```python
class BiomechanicalTags(BaseModel):
    ...
    torsional_joint_loading: bool = False  # NEW: pivoting / rotational loading under load
```

**Constraint addition**:
```python
class InjuryConstraint(BaseModel):
    ...
    block_torsional_loading: bool = False  # NEW
```

**Filter addition**:
```python
# 8. Torsional loading
if c.block_torsional_loading and ex.torsional_joint_loading:
    violations.append("torsional_loading_forbidden")
```

**Re-tagging required** for: `curtsy_lunge`, `kb_swing` (under fatigue), `russian_twist`,
any future medicine-ball rotational throws, golf-swing simulators, single-arm rows from
the floor.

#### 3.2.2 Spinal Shear vs Spinal Compression

**Risk**: `axial_compression_level` captures vertical load down the spine but NOT shear
forces from long moment arms (RDL, good morning). For users with spondylolisthesis
(spinal slippage), shear is the dangerous force, not compression.

The current prototype mitigates this implicitly by tagging RDLs and good mornings as
`axial_compression_level: HIGH` even though their *pure* compression is moderate. This
is a defensible patch but conflates two distinct risks.

**Two options**:

- **Option A (minimal change)**: Document the convention. Continue tagging hinge-pattern
  exercises as HIGH compression to capture their shear risk implicitly. Add a comment in
  schema.py explaining the convention. **Pros**: no schema change, no re-tagging.
  **Cons**: confusing for future tag auditors; we can't tell a spondy patient "you can
  do back squats but not RDLs" because both are tagged HIGH.

- **Option B (explicit field)**: Add `spinal_shear_level: ShearLevel` enum (NONE / MEDIUM /
  HIGH). Re-tag deadlifts, RDLs, good mornings, kettlebell swings, bent-over rows. **Pros**:
  intake LLM can express "no shear" for spondylolisthesis while still allowing axial
  compression. **Cons**: schema change + re-tagging.

**Recommendation**: Option B for clinical correctness. Effort cost: ~30 min re-tag on
existing 41 exercises. Spinal injuries are the highest-liability category — worth the
precision.

### 3.3 Architectural refinement: replace blunt `blocked_chains` with `JointAction`

**Bug**: `blocked_chains: [OPEN_UNLOADED]` for PFPS correctly blocks `leg_extension` but
**incorrectly blocks `lying_leg_curl` and `seated_leg_curl`** — open-chain knee flexion
(hamstring curl) is 100% safe and recommended for PFPS rehab. Our prototype throws out
the safe hamstring curl with the unsafe quadricep extension.

**Verification of bug** — re-running scenario 4 from the prototype:
- ❌ Currently `lying_leg_curl` is blocked under PFPS (incorrect — should be safe)
- ❌ Currently `seated_leg_curl` is blocked under PFPS (incorrect — should be safe)

**Solution**: replace `kinetic_chain_loading` + `blocked_chains` with a finer-grained
`joint_actions` enum that specifies BOTH the joint and the action.

```python
class JointAction(str, Enum):
    # Knee
    KNEE_EXTENSION_OPEN = "KNEE_EXTENSION_OPEN_CHAIN"     # leg extension machine
    KNEE_EXTENSION_CLOSED = "KNEE_EXTENSION_CLOSED_CHAIN" # squat ascent
    KNEE_FLEXION_OPEN = "KNEE_FLEXION_OPEN_CHAIN"          # leg curl machine
    KNEE_FLEXION_CLOSED = "KNEE_FLEXION_CLOSED_CHAIN"      # squat descent
    # Hip
    HIP_FLEXION_OPEN = "HIP_FLEXION_OPEN_CHAIN"            # lying leg raise
    HIP_FLEXION_CLOSED = "HIP_FLEXION_CLOSED_CHAIN"        # squat descent
    HIP_EXTENSION_CLOSED = "HIP_EXTENSION_CLOSED_CHAIN"    # squat ascent, hip thrust, deadlift
    # Spine
    SPINAL_FLEXION_DYNAMIC = "SPINAL_FLEXION_DYNAMIC"      # crunch
    SPINAL_EXTENSION_DYNAMIC = "SPINAL_EXTENSION_DYNAMIC"  # hyperextension
    SPINAL_ROTATION_DYNAMIC = "SPINAL_ROTATION_DYNAMIC"    # russian twist
    SPINAL_LATERAL_FLEXION_DYNAMIC = "SPINAL_LATERAL_FLEXION_DYNAMIC"
    SPINAL_ISOMETRIC_BRACING = "SPINAL_ISOMETRIC_BRACING"  # plank
    # Shoulder
    SHOULDER_OVERHEAD_LOADED = "SHOULDER_OVERHEAD_LOADED"
    SHOULDER_HORIZONTAL_LOADED = "SHOULDER_HORIZONTAL_LOADED"
    # Ankle
    ANKLE_PLANTARFLEXION_LOADED = "ANKLE_PLANTARFLEXION_LOADED"  # calf raise
```

Per exercise: `joint_actions: List[JointAction]`. For example:
- `leg_extension`: `[KNEE_EXTENSION_OPEN]`
- `lying_leg_curl`: `[KNEE_FLEXION_OPEN]`
- `back_squat`: `[KNEE_EXTENSION_CLOSED, KNEE_FLEXION_CLOSED, HIP_EXTENSION_CLOSED, HIP_FLEXION_CLOSED]`

PFPS constraint: `blocked_joint_actions: [KNEE_EXTENSION_OPEN]`. Now correctly blocks ONLY
leg extension, allowing leg curls and closed-chain squats. **This is the actual fix for
the bug.**

`kinetic_chain_loading` and `blocked_chains` can be **removed** after this refinement —
the `joint_actions` enum subsumes both (the chain status is encoded in each action).
Alternatively, keep `kinetic_chain_loading` as a redundant high-level summary for
debugging and remove `blocked_chains` from the constraint vector.

**Effort cost**: 1 day to define the enum (~30 entries) + re-tag the 41 prototype
exercises with their joint actions. Most exercises will have 1–3 actions.

---

## 4. Web Search Containment (Critical Architectural Boundary)

### 4.1 The risk

`app/agents/training_agent.py` has a 3-phase Adaptive RAG, where Phase 3 calls Tavily
for web-search fallback. Web search returns free-text content. If the LLM picks up
exercise names from this content (e.g., "Cossack squat," "Jefferson curl") and writes
them into the structured `workout` field, **those exercises were never tagged**. They
have no biomechanical safety profile. The deterministic filter never saw them.

This is the **single largest leak vector** in the otherwise-deterministic architecture.

### 4.2 The boundary rule

> **Web-sourced content MAY inform the LLM's coaching narrative (summary, tip, descriptive
> text). It MUST NOT introduce exercise identifiers into the structured `workout` field.**

### 4.3 The enforcement mechanism

Constrained decoding via `Literal[*safe_pool_ids]`. The LLM's structured output schema is
built dynamically per request:

```python
SafeExerciseId = Literal[tuple(e.exercise_id for e in safe_pool)]

class WorkoutItem(BaseModel):
    exercise_id: SafeExerciseId    # ← LLM physically cannot emit unsafe / untagged IDs
    day: str
    sets: int = Field(ge=1, le=8)
    reps: str
```

At token-decoding time, the model's probability mass for IDs outside `safe_pool` collapses
to zero. Even if web search told it about "Nordic Curl," `nordic_curl` is not a valid
`SafeExerciseId` enum member — the structured-output decoder cannot emit it.

The LLM is free to mention Nordic Curls in the `summary` / `tip` text fields. That's
narrative coaching, not prescription. The user reads it, learns about the exercise, and
can ask their coach. **But the structured workout the app actually serves them only
contains tagged-and-filtered IDs.**

### 4.4 Library expansion path (separate from real-time safety)

When web search consistently surfaces high-quality exercises that aren't in the library,
they deserve to be added — but **never in the user request path**. Async pipeline:

1. Background job logs all web-search exercise mentions across requests.
2. Weekly: top-N most-mentioned candidates are extracted.
3. Each candidate is run through the same biomechanical-tagging LLM that built the static
   library (chain-of-thought, structured output to `BiomechanicalTags` schema).
4. A physio reviews the auto-tag in a queue.
5. Approved candidates are added to the permanent library with full tags.
6. They become available for prescription on the next deployment.

**The user never gets an untagged exercise in real-time.** The library grows on a moderated
schedule.

### 4.5 What about training_agent.py's existing prompts?

The current system prompt explicitly tells the LLM to "discard the database items and use
your expert knowledge to generate proper exercises." That instruction is **incompatible**
with the Vetted Manifold architecture and must be **removed**.

New prompt direction: "Select exercises ONLY from the safe_pool provided in the structured
output schema. If safe_pool is too small for a balanced day, you have a `refuse: true`
boolean — use it." This is a meaningful prompt rewrite (~50 lines).

---

## 5. Keyword Matching Audit

> **Result**: zero string-keyword matching exists in the new safety pipeline. Detail below.

### 5.1 Where strings appear in `app/safety/`

| Location | Use | Risk |
|---|---|---|
| Enum values (`"KNEE"`, `"CLOSED_LOADED"`, etc.) | Pydantic enforces exact match | None — closed vocabulary, the LLM can only emit valid values or fail validation |
| `BiomechanicalTags.name` | Informational display only | None — never read by the filter |
| `BiomechanicalTags.exercise_id` | Opaque identifier in `Literal[...]` | None — exact identity match by enum |
| Intake LLM input (free-text injury) | Natural-language interpretation by gpt-4o-mini | **Soft underbelly** — see §5.3 |

### 5.2 What the filter does NOT do

```python
# THESE ARE EXAMPLES OF WHAT WE DELETED. None of these patterns exist anymore:
if "squat" in name.lower(): block()                          # ← false positive on "leg raise"
if "raise" in name.lower(): block()                          # ← false positive on "lying leg raise"
if re.search(r"\bbeef\b", text):                             # ← legacy diet code, separate concern
fuzzy_similarity(exercise_name, restricted_keyword) > 0.85   # ← never existed in safety
```

The filter operates on **enum set intersection** and **integer ordinal comparison**. There
is no substring matching, no fuzzy matching, no regex, no `.lower()`. The original
keyword-backstop class of bugs (synonym mismatch, substring collision, negation neglect)
is structurally impossible.

### 5.3 The remaining LLM boundary

The one place natural language enters the safety pipeline is `intake.py`, where free-text
injury descriptions are translated to `InjuryConstraint`. Pydantic + OpenAI structured
output is the safety boundary here:

- **Hallucinated body parts** → rejected at parse time (closed vocabulary).
- **Unknown injury terms** ("snapping hip syndrome," "labrum tear") → the LLM may guess
  wrong about which joints / actions to block.

Mitigations:

- Explicit injury → constraint mappings in the system prompt for common terms (see
  `intake.py` `_SYSTEM_PROMPT`).
- Conservative defaults (when LLM is uncertain, ceilings stay HIGH but specific axes
  the LLM identifies are tightened).
- Fail-safe fallback: any LLM error returns a maximally-restrictive constraint.
- Future: add a `physio_referral_recommended: bool` field that the LLM can set when
  the injury description is medically ambiguous — surfaces as refusal + physio referral.
- Future: shadow-mode evaluation where intake constraints are reviewed by a physio for
  a sample of real user inputs; misclassifications drive prompt refinement.

---

## 6. Refusal Threshold Strategy (Segment-Targeted Coverage)

### 6.1 Why "absolute count" is wrong

A naive threshold (`if len(safe_pool) < 5: refuse`) produces grotesque failure modes:

> User has knee injury + wrist pain → safe_pool = {crunch, russian_twist, plank, side_plank,
> dead_bug, clamshell}. Six items, threshold passes. System builds a "leg day" out of five
> core exercises and one clamshell. **Unsafe in practice** — that's not a leg day, it's a
> mockery of one. Worse: it conveys false confidence ("the system gave me a plan, so it
> must be safe to train") when the right answer was "see a physio."

### 6.2 Correct strategy: Muscle-segment coverage

Divide the exercise library into anatomical segments:

| Segment | Examples |
|---|---|
| Lower body anterior (quads, hip flexors) | Squats, lunges, leg extension, step-ups |
| Lower body posterior (glutes, hamstrings, calves) | RDLs, deadlifts, hip thrust, glute bridge, calf raise, leg curl |
| Lower body lateral (abductors, adductors) | Clamshell, lateral lunges, single-leg work |
| Core anterior (rectus, obliques) | Crunch, dead bug, hanging leg raise, plank |
| Core posterior (erectors, multifidus) | Hyperextension, bird-dog |
| Upper body push (chest, shoulders, triceps) | Bench, push-up, overhead press |
| Upper body pull (back, biceps) | Row, pull-up, lat pulldown |

Each tagged exercise is assigned 1 primary segment (and optionally 1–2 secondary segments).

### 6.3 The refusal algorithm

```python
def is_balanced_for_target(safe_pool, target_segments, min_per_segment=2):
    """
    target_segments is derived from the user query:
      - "leg day" → [lower_anterior, lower_posterior]
      - "full body" → [lower_anterior, lower_posterior, upper_push, upper_pull, core_anterior]
      - "push day" → [upper_push, core_anterior]
    """
    coverage = {}
    for ex in safe_pool:
        coverage[ex.primary_segment] = coverage.get(ex.primary_segment, 0) + 1
    return all(coverage.get(seg, 0) >= min_per_segment for seg in target_segments)


def maybe_refuse(safe_pool, target_segments) -> Optional[RefusalReason]:
    if is_balanced_for_target(safe_pool, target_segments):
        return None
    insufficient = [
        seg for seg in target_segments
        if sum(1 for ex in safe_pool if ex.primary_segment == seg) < 2
    ]
    return RefusalReason(
        cause="insufficient_segment_coverage",
        missing_segments=insufficient,
        message=(
            "Based on the injuries you reported, a safe and balanced "
            f"{', '.join(target_segments)} workout is not biomechanically "
            "possible right now. We strongly recommend consulting a "
            "physiotherapist, or focusing on upper-body active recovery instead."
        ),
    )
```

### 6.4 Schema addition

```python
class Segment(str, Enum):
    LOWER_ANTERIOR = "LOWER_ANTERIOR"
    LOWER_POSTERIOR = "LOWER_POSTERIOR"
    LOWER_LATERAL = "LOWER_LATERAL"
    CORE_ANTERIOR = "CORE_ANTERIOR"
    CORE_POSTERIOR = "CORE_POSTERIOR"
    UPPER_PUSH = "UPPER_PUSH"
    UPPER_PULL = "UPPER_PULL"

class BiomechanicalTags(BaseModel):
    ...
    primary_segment: Segment
    secondary_segments: List[Segment] = []
```

Re-tagging effort on the prototype 41 exercises: ~10 minutes.

---

## 7. Phased Implementation Roadmap

| Phase | Effort | Deliverable | Gate / Definition of Done |
|---|---|---|---|
| **0. Prototype** ✅ | done | `app/safety/{schema,filter,intake}.py` + 41 tagged exercises + 5 validated scenarios | Done |
| **1. Senior review tagging fixes** | 0.5 d | Update `tags_lower_body.json`: planks tag LUMBAR_SPINE, lying_leg_raise tags LUMBAR_SPINE, hip_thrust grip LIGHT. Re-run scenario 3 (disc herniation) — planks and lying leg raise must now appear in BLOCKED list. | Disc herniation scenario blocks planks (was leak in v0) |
| **2. Schema additions: torsional + shear** | 1 d | Add `torsional_joint_loading: bool` and `spinal_shear_level: ShearLevel` to BiomechanicalTags. Add parallels to InjuryConstraint. Update filter to 9 checks. Re-tag 41 exercises. | New scenarios: ACL recovery → curtsy_lunge blocked, reverse_lunge allowed. Spondylolisthesis → RDL blocked by shear, back squat allowed (compression only). |
| **3. JointAction refinement** | 1.5 d | Add `JointAction` enum (~30 entries). Add `joint_actions: List[JointAction]` to BiomechanicalTags. Add `blocked_joint_actions` to InjuryConstraint. Re-tag 41 exercises. Remove `blocked_chains` (subsumed). Re-run PFPS scenario. | PFPS scenario: leg_extension blocked, lying_leg_curl ALLOWED, seated_leg_curl ALLOWED. (Was the original bug.) |
| **4. Web-search containment** | 0.5 d | Document the boundary rule. Build constrained-output schema generator that takes safe_pool → `Literal[*ids]`. Smoke test: ensure no untagged exercise can appear in `workout` field even when explicitly mentioned in web context. | LLM with explicit prompt "include nordic curl" + web context mentioning it → structured output still uses only safe_pool IDs. |
| **5. Refusal-threshold implementation** | 0.5 d | Add `Segment` enum + `primary_segment` to tags. Implement `maybe_refuse(safe_pool, target_segments)` with segment coverage check. Wire into the workout-planner LLM (it sees a `refusal_reason` field it must propagate). | Groin + ankle scenario (Scenario 2) for "leg day" target → returns refusal, not 9-exercise core mockery. |
| **6. LLM-as-judge evaluation harness** | 1 d | `eval.py`: N=20 (injury × goal) pairs. Run through OLD pipeline (Three-Tier + Excluder) and NEW pipeline. GPT-4o (or Claude) as neutral judge. Score: Safety 1–10, Volume Capping 1–10, Split Balance 1–10, Media Integrity 1–10. CSV output. | Same input → same scores ±0.5 across re-runs (judge stability check). |
| **7. Run the benchmark** | 0.25 d | Comparison CSV: old vs new pipeline scores across 20 scenarios. | New Safety ≥ Old Safety + 1.0 with no regression > 0.5 on other axes. If gate fails → debug + iterate on tags, not architecture. |
| **8. Integrate into TrainingAgent** | 1.5 d | Rewrite `app/agents/training_agent.py`: (a) call intake → constraint, (b) load tagged candidates, (c) apply filter, (d) build `Literal[*safe_pool_ids]` output schema, (e) wire refusal branch, (f) remove "MASTER TRAINER DISCARD" prompt instruction. | End-to-end run on 5 manual injury queries returns safe workouts AND refuses gracefully for impossible cases. |
| **9. Retire legacy safety layers** | 0.5 d | Delete: Dynamic Excluder LLM call, Tier-3 LLM safety review, Python keyword backstop, fallback list of curated exercises. | All 5 prototype scenarios still safe; no references to deleted code remain. |
| **10. Expand tagging to full library** | 2 d | Auto-tagging pipeline: each existing exercise → biomechanical-tagging LLM with chain-of-thought → BiomechanicalTags. Manual audit of 100 random samples by team engineer + 50 by consulting physio. | Audit accuracy ≥ 95% engineer, ≥ 98% physio. |
| **11. Web-search expansion pipeline** | 0.5 d | Background job: log web-search exercise mentions, weekly top-N extraction, auto-tag, physio review queue. | First batch reviewed and approved within 1 week of deployment. |
| **12. Production rollout** | 0.5 d | Feature-flag the new pipeline. 10% canary → 50% → 100%. Monitor: safety incidents reported, refusal rate, average safe_pool size, user satisfaction by injury cohort. | No safety regressions in canary; refusal rate within expected band (10–20% for users with declared injuries, < 1% for healthy users). |
| **Total** | **~10 days** | | |

---

## 8. Verification & Testing Strategy

### 8.1 Unit tests (`app/safety/tests/`)

For each scenario in §6 of `README.md` plus the new ones added in phases 1–5:

```python
def test_pfps_blocks_leg_extension_allows_leg_curl():
    candidates = load_tags("tags_lower_body.json")
    constraint = InjuryConstraint(blocked_joint_actions=[JointAction.KNEE_EXTENSION_OPEN])
    safe = filter_safe_exercises(candidates, constraint)
    safe_ids = {e.exercise_id for e in safe}
    assert "leg_extension" not in safe_ids
    assert "lying_leg_curl" in safe_ids        # was bug in v0
    assert "seated_leg_curl" in safe_ids       # was bug in v0
    assert "back_squat" in safe_ids            # closed-chain rehab
```

Coverage: every scenario × every tagging fix must have at least one assertion.

### 8.2 LLM-as-judge regression suite

Phase 6 deliverable. After every tag change, re-run the eval. Any score drop > 0.5 on
Safety must be investigated before merge.

### 8.3 Shadow-mode comparison

Phases 8–9 only: run the new pipeline alongside the old one for 1 week. Both produce
workout plans; only the OLD plan is served to users. Compare the two via the LLM judge.
Identify cases where the new system would have blocked something the old system served
(real safety wins) and cases where the new system refuses but old system shipped a
working plan (potential over-restriction).

### 8.4 Physio audit

Phase 10 gate: 50 randomly-sampled auto-tagged exercises reviewed by a sports
physiotherapist. Accept rate ≥ 98% before library expansion is considered complete.

---

## 9. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Intake LLM misclassifies unusual injury terminology | Medium | High | Explicit mapping table in prompt; conservative defaults; fail-safe fallback; physio-referral flag |
| 2 | Tag-quality cascade — one mis-tag creates a leak across all queries hitting that exercise | Medium | High | Vector-similarity audit (cosine of exercise embedding vs known-safe/known-unsafe centroids); physio review on samples; LLM-as-judge regression |
| 3 | Web-search-induced exercise names bypass safety | Low (post-fix) | Critical | Constrained decoding `Literal[*safe_pool_ids]` enforces boundary at token level; prompt rewrite removes "discard DB, use expert knowledge" instruction |
| 4 | Refusal rate too high — users abandoned | Medium | Medium | Segment-coverage threshold is calibrated, not absolute count; tunable per-segment minimum; A/B test refusal copy |
| 5 | Refusal rate too low — unsafe plans served despite constraint | Low (post-fix) | Critical | Belt-and-suspenders: filter + constrained decoding + refusal threshold all enforce |
| 6 | LLM API rate limits / outages affect intake | Low | Medium | Cache common constraints by injury-string hash; fail-safe fallback returns restrictive constraint |
| 7 | Physio not available for audit during rollout | Medium | Medium | Engineer self-audit on samples first; clinical sign-off as a parallel track, not a blocker for canary |
| 8 | Production user reports actual injury from a served plan | Low | Critical | Logging: every workout served includes the constraint that produced it + safe_pool size. Incident playbook: immediate canary rollback. |

---

## 10. Production Rollout Plan

### 10.1 Pre-launch checklist

- [ ] All phase 1–9 gates passed.
- [ ] Full library tagged (phase 10).
- [ ] Physio sign-off on tagging accuracy.
- [ ] Refusal copy reviewed by physiotherapist + product.
- [ ] Logging: per-request constraint, safe_pool size, refusal cause, served exercise IDs.
- [ ] Incident playbook: how to roll back, how to triage a reported injury, escalation path.
- [ ] Web-search expansion pipeline running in shadow mode.

### 10.2 Canary phases

| Phase | % traffic | Duration | Gate to next |
|---|---|---|---|
| 10% canary | 10% | 7 days | No safety incidents reported; refusal rate within expected band; satisfaction not down |
| 50% expansion | 50% | 7 days | Same gates |
| 100% rollout | 100% | — | Same |

### 10.3 Monitoring dashboards

- **Safety**: incidents per 1k workouts served (target: 0)
- **Refusal rate**: by injury type, by goal — alert if jumps > 2× baseline
- **Safe pool size distribution**: histogram — alert if median drops > 30%
- **Intake LLM latency**: p50/p95 — alert if p95 > 2s
- **User satisfaction**: thumbs-up rate on workout responses for users with declared injuries
- **Library coverage**: % of intent-classified injuries with sufficient safe pool

---

## 11. Future Work (Post-v1)

Items deliberately deferred from v1 to keep scope shippable:

- **Severity tiers**: explicit `Severity` enum (ACUTE / SUBACUTE / CHRONIC_MANAGED /
  HISTORICAL) modulating constraint strictness. Currently handled implicitly by intake LLM.
- **Left/right granularity**: per-side injury → per-side exercise blocking. Required for
  asymmetric rehab programs.
- **ROM (range-of-motion) precision**: knee_flexion_deg ranges per exercise vs per
  injury. Required for post-op users with limited mobility.
- **Multi-condition composition rules**: "user has condition A AND condition B" might
  warrant constraint combinations the intake LLM doesn't currently handle. Worth offline
  analysis once we have real data on common combinations.
- **Specialized intake model**: fine-tune a small model on injury → constraint pairs
  drawn from physiotherapy literature + production logs.
- **Vector-similarity safety audit (offline only)**: cosine distance between exercise
  embeddings and known-unsafe/known-safe exemplars as a tag-quality QA tool. NOT a
  runtime safety layer.
- **Pre-workout safety questionnaire UI**: ask the user directly about specific
  contraindications instead of parsing free-text injuries.

---

## 12. Cross-Verification Prompt for Reviewing AI

If sharing this plan with a second AI for independent review, the most useful questions:

1. Does the 9-feature schema (after additions in §3.2) capture all clinically-significant
   biomechanical risks for orthopedic injuries? What's missing?
2. Is the `JointAction` enum vocabulary complete for lower body + core? Which actions
   are we missing that map to common injuries?
3. Spot-check 5–10 of the updated tags (after §3.1 fixes are applied) — any leaks?
4. Is the web-search containment rule sufficient, or are there channels we haven't
   identified through which an untagged exercise could reach the user?
5. Is the segment-coverage refusal threshold correctly designed, or should it incorporate
   additional dimensions (e.g., metabolic balance, push/pull ratio)?
6. What's the worst single failure mode of the intake LLM that the prompt + Pydantic
   schema can't catch?
