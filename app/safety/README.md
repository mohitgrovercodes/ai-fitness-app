# Biomechanical Safety Prototype — Lower Body + Core Slice

> **Purpose**: Validate the 7-Feature Biomechanical Tagging strategy on a
> controlled 41-exercise slice before rolling out across the full library.
> Self-contained — share this single document with any AI for cross-verification.

---

## 1. Architecture in One Picture

```
User input ("severe groin pull + ankle sprain")
        │
        ▼  [Intake LLM, ONE structured call]
InjuryConstraint  (Pydantic, closed vocabularies)
        │
        ▼  [Pure Python, deterministic, no LLM]
filter_safe_exercises(candidates, constraint) → safe_pool
        │
        ├── len(safe_pool) < threshold → REFUSE + recommend physio
        │
        ▼
LLM workout planner with structured output:
   exercise_id: Literal[*safe_pool_ids]   ← constrained decoding
        │
        ▼
Workout plan referencing only safe exercise IDs by reference
        │
        ▼  [Server-side join]
Final response with names, GIFs, sets/reps populated from canonical DB
```

The **only** LLM call inside the safety boundary is the intake translator,
and it's locked to a Pydantic schema that physically can't emit hallucinated
body regions or constraint values. Everything downstream is deterministic.

---

## 2. The 7 Features (Per-Exercise Tags)

Each exercise in the library carries this 7-tuple. Closed vocabularies on every
axis. Ordinal where ordering is meaningful, set/categorical otherwise.

| # | Feature | Type | Values | What it captures |
|---|---|---|---|---|
| 1 | `primary_joints_involved` | **set** of `Joint` | HIP, KNEE, ANKLE, LUMBAR_SPINE, THORACIC_SPINE, CERVICAL_SPINE, SHOULDER, ELBOW, WRIST | Joints that act as **prime movers under load**. Stabilizing joints (e.g. spine during a deadlift) are NOT included here — their risk is captured by feature 3. |
| 2 | `kinetic_chain_loading` | single `ChainStatus` | CLOSED_LOADED, CLOSED_SUPPORTED, OPEN_UNLOADED | Whether the moving limb is fixed against resistance, fixed but supported, or moves freely. |
| 3 | `axial_compression_level` | ordinal `CompressionLevel` | NONE(0), MEDIUM(1), HIGH(2) | Compressive force down the spine. Bar-on-back squats → HIGH, dumbbells at sides → MEDIUM, supine → NONE. |
| 4 | `grip_requirement` | ordinal `GripDemand` | NONE(0), LIGHT(1), HEAVY(2) | Grip force the exercise demands. Deadlifts/hanging → HEAVY, dumbbell hold → LIGHT, machine → NONE. |
| 5 | `joint_impact_level` | ordinal `ImpactLevel` | NONE(0), LOW(1), HIGH(2) | Ballistic/plyometric impact. Box jumps → HIGH, KB swing → LOW, controlled lifts → NONE. |
| 6 | `upper_limb_stabilization` | binary `UpperLimbDemand` | NONE, ACTIVE | Whether the upper limbs bear bodyweight. Plank/push-up/hanging → ACTIVE, lower body lifts → NONE. |
| 7 | `metabolic_density` | ordinal `MetabolicDensity` | LOW(0), MEDIUM(1), HIGH(2) | CV/respiratory demand. Circuits/sprints → HIGH, compound w/ rest → MEDIUM, isolation → LOW. |

---

## 3. The Constraint Vector (Parallel to the 7 Features)

The intake LLM populates this from raw injury text. Defaults are the **most permissive**
values (HIGH ceilings, empty block lists) so an absent constraint = "no restriction
on that axis."

| Field | Type | Default | Parallels Feature |
|---|---|---|---|
| `blocked_joints` | set of Joint | `[]` | 1 — exercises with any of these in `primary_joints_involved` are filtered |
| `blocked_chains` ⭐ | set of ChainStatus | `[]` | 2 — exercises with this chain status are filtered |
| `max_axial_compression` | CompressionLevel | HIGH | 3 — exercises exceeding this level are filtered |
| `max_grip_requirement` | GripDemand | HEAVY | 4 — exercises exceeding this are filtered |
| `max_impact` | ImpactLevel | HIGH | 5 — exercises exceeding this are filtered |
| `block_upper_limb_active` | bool | False | 6 — when True, ACTIVE upper-limb exercises are filtered |
| `max_metabolic_density` | MetabolicDensity | HIGH | 7 — exercises exceeding this are filtered |

### ⭐ The one modification to your original 7-feature plan

I added `blocked_chains: List[ChainStatus]`. Rationale:

**Patellofemoral pain syndrome (PFPS)** is the textbook case where chain status matters
beyond joint blocking:
- **Open-chain knee extension** (leg extension machine) is the classic aggravator
  and must be blocked.
- **Closed-chain knee loading** (squats, leg press) is actually used as *rehab*
  for PFPS — blocking it would be incorrect.

Without this modification, the only way to block leg extension would be
`blocked_joints: [KNEE]`, which also incorrectly blocks closed-chain squats.
The modification keeps the 7-feature schema unchanged on the *exercise* side
(no extra tagging burden) and adds one constraint axis to express finer-grained
filtering. **Validated in Scenario 4 below.**

---

## 4. The Filter (Pure Python, Deterministic)

```python
def safety_violations(ex: BiomechanicalTags, c: InjuryConstraint) -> List[str]:
    v = []
    # 1. Joint loading
    if set(ex.primary_joints_involved) & set(c.blocked_joints): v.append("blocked_joint")
    # 2. Chain status
    if ex.kinetic_chain_loading in c.blocked_chains: v.append("blocked_chain")
    # 3. Axial compression (ordinal)
    if ex.axial_compression_level > c.max_axial_compression: v.append("axial_compression_exceeded")
    # 4. Grip (ordinal)
    if ex.grip_requirement > c.max_grip_requirement: v.append("grip_exceeded")
    # 5. Impact (ordinal)
    if ex.joint_impact_level > c.max_impact: v.append("impact_exceeded")
    # 6. Upper-limb stabilization
    if c.block_upper_limb_active and ex.upper_limb_stabilization == ACTIVE: v.append("upper_limb_loading_forbidden")
    # 7. Metabolic density (ordinal)
    if ex.metabolic_density > c.max_metabolic_density: v.append("metabolic_density_exceeded")
    return v

is_safe(ex, c) := safety_violations(ex, c) == []
```

7 checks, ~microseconds per exercise, fully auditable (each violation is named).

---

## 5. The 41-Exercise Prototype Slice

Categories covered (intentional coverage of high-risk and low-risk ends):

| Pattern | Count | Example IDs |
|---|---|---|
| Squat variants | 6 | back_squat, front_squat, goblet_squat, bw_squat, box_squat, wall_sit |
| Hinge variants | 5 | rdl_bb, conv_deadlift, sumo_deadlift, good_morning, kb_swing |
| Lunge / step | 5 | walking_lunge, reverse_lunge, bulgarian_split_squat, step_up, curtsy_lunge |
| Posterior chain isolation | 5 | hip_thrust, glute_bridge, single_leg_glute_bridge, hyperextension, cable_pull_through |
| Knee isolation | 4 | leg_extension, lying_leg_curl, seated_leg_curl, leg_press |
| Calf | 3 | standing_calf_raise, seated_calf_raise, donkey_calf_raise |
| Core | 8 | plank, side_plank, crunch, russian_twist, hanging_leg_raise, lying_leg_raise, dead_bug, bird_dog |
| Rehab / mobility | 5 | pallof_press, clamshell, sl_rdl_bw, couch_stretch, 90_90_hip_mobility |

Full tagged JSON lives in `tags_lower_body.json`. Sample rows (numeric values are
the ordinal IntEnum integers):

| ID | Joints | Chain | AxialComp | Grip | Impact | UpperLimb | MetDens |
|---|---|---|---|---|---|---|---|
| back_squat | HIP, KNEE, ANKLE | CLOSED_LOADED | 2 | 1 | 0 | NONE | 1 |
| rdl_bb | HIP | CLOSED_LOADED | 2 | 2 | 0 | NONE | 1 |
| leg_press | HIP, KNEE, ANKLE | CLOSED_SUPPORTED | 0 | 0 | 0 | NONE | 1 |
| leg_extension | KNEE | OPEN_UNLOADED | 0 | 0 | 0 | NONE | 0 |
| plank | (none — isometric) | CLOSED_LOADED | 0 | 0 | 0 | ACTIVE | 0 |
| lying_leg_raise | HIP | OPEN_UNLOADED | 0 | 0 | 0 | NONE | 0 |
| kb_swing | HIP | CLOSED_LOADED | 1 | 2 | 1 | NONE | 2 |

### Tagging conventions that matter

- **"Prime mover under load" is the rule for `primary_joints_involved`.** A
  back squat involves the spine isometrically — it's not a prime mover, so
  LUMBAR_SPINE is NOT in the list. Spinal risk is caught by `axial_compression_level: HIGH`.
- **Plank has `primary_joints_involved: []`** — it's isometric, no joint is a
  prime mover. The upper-body demand is captured by `upper_limb_stabilization: ACTIVE`.
- **Hip thrust has `axial_compression_level: NONE`** — load is across the
  pelvis with the upper back braced on a bench, no compression down the spine.

---

## 6. Validation Scenarios (Already Run)

Filter executed against the 41-exercise pool. Results verified manually:

### Scenario 1 — Mild knee pain
`InjuryConstraint(blocked_joints=[KNEE], max_impact=NONE)`
- ✅ SAFE (23): RDLs, hinges, hip thrusts, calf work, all core, mobility
- ❌ BLOCKED (18): all squats, lunges, step-ups, leg extension, leg curls, leg press, kettlebell swing

### Scenario 2 — Severe groin pull + ankle sprain
`InjuryConstraint(blocked_joints=[HIP, ANKLE], max_impact=NONE)`
- ✅ SAFE (9): wall_sit, leg_extension, leg_curls (KNEE-only), plank, side_plank, crunch, russian_twist, pallof_press
- ❌ BLOCKED (32): everything hip-loading and everything ankle-loading
- **Refusal trigger candidate** — 9 exercises is below realistic threshold for a
  balanced session. The integrated planner should surface "consult a physio" here.

### Scenario 3 — Lumbar disc herniation L4-L5
`InjuryConstraint(blocked_joints=[LUMBAR_SPINE], max_axial_compression=NONE, max_impact=NONE)`
- ✅ SAFE (21): bw_squat, wall_sit, hip thrust, glute bridges, leg press, machines, plank, side plank, supine core, mobility
- ❌ BLOCKED (20): all axially-loaded squats/deadlifts/hinges, all dynamic spinal flexion (crunch, russian twist, hanging leg raise), hyperextension
- Healthy pool — system can still build a meaningful day with zero spinal load.

### Scenario 4 — Patellofemoral pain syndrome ⭐ (validates the modification)
`InjuryConstraint(blocked_chains=[OPEN_UNLOADED], max_impact=NONE)`
- ✅ SAFE (32): **all squats, lunges, leg press, deadlifts, hip thrust** — closed-chain knee loading IS rehab
- ❌ BLOCKED (9): **leg_extension, leg_curls** (OPEN_UNLOADED, the textbook aggravators), kb_swing (impact), hanging/lying leg raise, bird-dog, pallof press, clamshell
- **This is the test the original 7-feature plan couldn't pass cleanly.**
  Without `blocked_chains`, you'd have to use `blocked_joints: [KNEE]` and lose
  the closed-chain rehab options.

### Scenario 5 — Wrist sprain
`InjuryConstraint(blocked_joints=[WRIST], max_grip_requirement=NONE, block_upper_limb_active=True)`
- ✅ SAFE (19): bw_squat, wall_sit, hip thrust, glute bridges, leg press, machines, lying_leg_raise, dead_bug, clamshell, mobility
- ❌ BLOCKED (22): all deadlifts and grip-heavy exercises, all plank variants, hanging exercises, bird-dog, pallof press
- Tagging-precision note: `back_squat` is blocked here because we tagged
  `grip_requirement: LIGHT` (you stabilize the bar with hands). Defensible —
  even minor wrist articulation under load is risky with a fresh sprain — but
  this is exactly the kind of judgement call the LLM-as-judge eval should surface
  for iteration.

---

## 7. Files in This Prototype

| File | Purpose | Size |
|---|---|---|
| `schema.py` | Pydantic enums + `BiomechanicalTags` + `InjuryConstraint` | ~150 lines |
| `filter.py` | `safety_violations`, `is_safe`, `filter_safe_exercises`, `filter_with_audit` | ~75 lines |
| `intake.py` | LLM-driven injury → constraint translator with fail-safe fallback | ~90 lines |
| `tags_lower_body.json` | 41 tagged exercises | ~45 lines |
| `README.md` | This document | — |

---

## 8. Implementation Plan (To Full Rollout)

| Step | Effort | Deliverable | Gate |
|---|---|---|---|
| **0. ✅ Schema + filter + slice tagged + 5 scenarios validated** | — | Done in this prototype | — |
| **1. Wire intake LLM end-to-end** | 0.5 d | Calling `translate_injury_to_constraint("real injury text")` returns a sensible InjuryConstraint | Manual sanity-check on 10 injury strings |
| **2. Build LLM-as-judge evaluation harness** | 1 d | `eval.py`: runs N=20 (injury, goal) pairs through both pipelines, GPT-4o or Claude as judge, scores on Safety / Volume / Split Balance / Media Integrity; outputs CSV | Eval reproducibility — same input → same scores ±0.5 |
| **3. Run the benchmark** | 0.25 d | Comparison CSV: old (Three-Tier+Excluder) vs new (Vetted Manifold) on the slice | New Safety ≥ Old Safety + 1.0 with no regression on Balance |
| **4. Integrate into TrainingAgent** | 1 d | `training_agent.py` modified: intake → filter → `Literal[*safe_pool_ids]` constrained output → server-side join. Refusal branch when `len(safe_pool) < threshold`. | End-to-end run on 5 injury queries returns vetted plans |
| **5. Retire legacy safety layers** | 0.5 d | Remove Dynamic Excluder, Tier-3 LLM safety pass, Python keyword backstop. Existing prompts cleaned. | All 5 scenarios still produce safe plans |
| **6. Expand tagging beyond slice** | 2 d | Tag remaining ~2,800 exercises via auto-LLM pipeline w/ chain-of-thought, manual audit of 100 samples | Audit accuracy ≥ 95% on samples |
| **7. Production rollout** | 0.5 d | Feature-flag the new pipeline, route 10% → 50% → 100% of traffic | Safety regression rate < baseline |
| **Total** | ~6 days | | |

---

## 9. Known Limitations of This v1 (Honest Disclosure)

These are intentional simplifications to ship a prototype. Surface in eval, decide
whether to fix or accept.

- **No severity tiers.** "Mild knee tweak" and "acute meniscus tear" both map to
  `blocked_joints: [KNEE]`. The intake LLM can compensate (more restrictive
  thresholds for acute), but it's implicit. Could add explicit `Severity` enum
  if needed.
- **No left/right granularity.** A left-side groin pull blocks all hip-loading,
  including right-leg unilateral work that's actually safe. Practical for v1,
  may matter for asymmetric injuries later.
- **No ROM precision.** A partial squat (60° knee flexion) is one exercise tag,
  a full deep squat is another. Exercise-granularity rather than angle-granularity.
- **Tagging-precision cascade**: a single mis-tag (e.g. setting
  `axial_compression_level: NONE` on something that actually compresses) creates
  a leak. Vector-similarity audit (cosine between exercise embeddings and
  category exemplars) is a planned QA tool, not a runtime safety layer.
- **Static fallback for intake LLM failure**: when the LLM call errors, intake
  returns a maximally-restrictive constraint. Better to over-restrict than leak,
  but the user gets a degraded experience.

---

## 10. Cross-Verification Prompt for Another AI

If you're sharing this with a second AI for review, the questions worth asking are:

1. **Is the 7-feature axis system genuinely orthogonal?** Are there real
   biomechanical risks the 7 features fail to capture?
2. **Is the `blocked_chains` modification justified?** Are there other
   single-condition cases that demand a similar finer-grained block on a feature
   that's currently only a per-exercise tag?
3. **Are the 41 example tags accurate?** Spot-check 5–10 exercises. Misclassifications
   are the only way the architecture leaks.
4. **Do the 5 validation scenarios cover the realistic failure space?** What's
   the worst test case we're missing?
5. **Is the refusal threshold strategy correct?** Should the system refuse based on
   absolute count, balanced muscle group coverage, or something else?
