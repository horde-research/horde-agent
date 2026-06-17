# Agent Implementation Plan

This document assumes the current repo already has a useful workflow-oriented execution layer and that the next step is to add a real agentic controller on top of it.

The goal is not to replace the current tools with "agent magic". The goal is to build a constrained, inspectable, stateful controller that can:

- decide what stage to run next
- evaluate stage quality using explicit rules
- retry or adjust parameters when quality is insufficient
- stop when success criteria are met or recovery is no longer worth it

This plan is intentionally biased toward a reliable first version, not the most autonomous version.

## Agreed V1 Direction

The first version should be a single-agent, constrained controller over the known `full` pipeline graph. The existing workflow remains the plan of record; the agent follows it and only iterates on stages whose quality gates require recovery.

V1 decisions:

- Finish goal: `success = required_artifacts_exist AND required_quality_gates_pass AND no_unresolved_blockers`
- Scope: `full` mode only
- Quality model: hybrid deterministic rules plus bounded LLM judgment
- Recovery: bounded retries and bounded config changes
- Resume: allowed, but ask the user before skipping stages that were executed in a previous run
- Budget model: no explicit budget enforcement in v1
- Observability: LangSmith is the required trace backend for run trajectory and stage-level spans
- Agent shape: single-agent controller, not multi-agent
- Non-goal: open-ended autonomy; v1 stays on a known graph

## Part 1: Open Questions to Define Before Implementation

This section lists the questions that should be answered before building the agent. The questions below now include the agreed v1 decision and the baseline rationale that supports it.

### 1. What is the exact finish goal of one agent run?

Why this matters:
If the finish goal is vague, the agent cannot know when to stop. It will either terminate too early or loop forever.

Decision:
Use this success contract:

- `success = required_artifacts_exist AND required_quality_gates_pass AND no_unresolved_blockers`

Baseline rationale:
One successful run should produce:

- a usable training dataset artifact, usually `sft.jsonl`
- a dataset summary and validation report
- a trained adapter artifact, if training is enabled
- evaluation artifacts, including predictions and failure analysis
- a final report artifact

The run should count as successful only if all enabled stages meet their minimum quality thresholds.

### 2. Should the agent own the whole pipeline or only the optimization loop around it?

Why this matters:
This decides whether the agent is a full controller or just a planner that wraps the workflow.

Decision:
The workflow remains the plan. The agent follows that plan and iterates only on stages where quality requires another attempt or adjusted parameters.

Baseline rationale:
The first version should own orchestration decisions around the pipeline, but not replace tool internals.

That means:

- tools remain deterministic execution units
- the agent decides when to call each tool, with what config, and whether to retry
- the agent does not implement crawling, annotation, training, or eval itself

### 3. What is the unit of action?

Why this matters:
The agent needs a finite action space. "Use any tool however you want" is too open for a reliable v1.

Decision:
Use the proposed stage-level units of action.

Baseline rationale:
The action unit should be one stage-level tool invocation, not an arbitrary low-level operation.

Recommended initial action set:

- `generate_taxonomy`
- `collect_data`
- `build_sft_dataset`
- `build_dataset`
- `train_model`
- `evaluate_model`
- `generate_report`
- `retry_stage`
- `adjust_stage_config`
- `stop_success`
- `stop_failure`

In practice, `retry_stage` and `adjust_stage_config` can resolve to rerunning one of the main stage actions with changed config.

### 4. Which stages are required and which are optional?

Why this matters:
The agent needs to know if skipping a stage is legal.

Decision:
V1 supports `full` mode only.

Baseline rationale:
The first version should keep one required stage sequence:

- `full`: taxonomy -> collection -> SFT build -> dataset build -> train -> eval -> report

Optional stages and other modes can be added after the controller is proven on `full`.

### 5. What quality gates exist at each stage?

Why this matters:
This is the core of the agent. Without quality gates it is just a tool caller.

Decision:
Use the proposed initial stage-level gates.

Baseline rationale:
Each stage should return a `QualityReport` with:

- `pass_fail`
- `score`
- `blocking_issues`
- `warnings`
- `metrics`
- `recommended_actions`

Initial stage-level gates:

#### Taxonomy

- non-empty category list
- non-empty subcategories for enough categories
- non-empty query list
- duplicate rate below threshold
- multilingual query coverage above threshold

#### Collection

- minimum sample count
- minimum average text length
- duplicate rate below threshold
- bad-domain leakage below threshold
- empty-text rate below threshold

#### SFT build

- minimum example count
- schema-valid example rate above threshold
- language consistency above threshold
- malformed chat-format rate below threshold

#### Dataset build

- expected columns present
- minimum sample count
- modality inferred consistently
- validation warnings below blocking threshold

#### Training

- training finished without crash
- adapter artifact exists
- metrics file exists
- loss is not obviously degenerate
- no strong instability markers in logs

#### Evaluation

- predictions artifact exists
- failure rate below threshold or explainable
- repetition rate below threshold
- cluster analysis completed successfully

#### Final report

- all expected sections present
- all required artifact references resolvable

### 6. Should quality be purely deterministic, purely LLM-based, or hybrid?

Why this matters:
Pure LLM judgment is expensive and unstable. Pure rules miss nuanced quality failures.

Decision:
Use a hybrid quality model.

Baseline rationale:
Use a hybrid model with deterministic rules first and LLM judgment only where semantic interpretation is genuinely needed.

Recommended policy:

- deterministic validators decide pass/fail wherever possible
- LLM-based judges provide structured recommendations, explanations, or tie-breaks
- LLMs should not be the only mechanism that can declare success

### 7. What should be considered a blocking failure versus a recoverable failure?

Why this matters:
The agent needs to know whether to retry, adapt, or stop.

Decision:
Use the proposed blocker/recoverable/warning classification.

Baseline rationale:
Treat failures in three classes:

#### Hard blockers

- missing required API key
- missing required dependency
- missing required input artifact
- tool contract violation
- zero usable output after maximum retries

#### Recoverable failures

- low sample count
- low multilingual coverage
- too many malformed examples
- weak eval metrics
- unstable training run with adjustable config

#### Soft warnings

- low diversity
- low domain spread
- small eval sample
- non-critical reporting issues

### 8. How much autonomy should the agent have when changing configs?

Why this matters:
Too much freedom will create irreproducible behavior. Too little freedom prevents recovery.

Decision:
Use bounded config changes as proposed.

Baseline rationale:
Config changes should be bounded by explicit policy.

Suggested allowed adjustments in v1:

- change `max_queries`
- change collection concurrency or result limits
- change SFT batch size and delay
- change training `max_steps`
- change training batch size
- change grad accumulation
- change learning rate within a bounded multiplier
- switch among approved LoRA presets
- reduce sample count for recovery/debug runs

Suggested forbidden adjustments in v1:

- inventing new tools
- changing model family without approval
- changing run mode midstream
- editing prompts dynamically unless explicitly enabled

### 9. What are the retry rules?

Why this matters:
Without retry limits the agent can spin forever.

Decision:
Use the proposed retry limits for v1.

Baseline rationale:
Define both global and per-stage retry limits.

Suggested baseline:

- per-stage retry limit: 2
- global decision loop limit: 20 actions
- training retry limit: 2
- collection retry limit: 2
- eval retry limit: 1 unless upstream training changed

Retry should always require a reason and a proposed config delta.

### 10. What is the budget model?

Why this matters:
Agentic systems need explicit constraints on cost and elapsed time.

Decision:
Do not implement an explicit budget model in v1.

Baseline rationale:
Budget tracking is useful, but it should not block the first implementation. Retry limits and graph-step limits are enough for v1.

Future budget fields can track:

- wall-clock time
- LLM call count
- stage retry counts
- rough token or request budgets

The first version should still keep retry limits, but cost and time budgets can be added later.

### 11. Should the agent be allowed to skip ahead if current artifacts already exist?

Why this matters:
This affects resumability and real-world ergonomics.

Decision:
Yes, but ask the user before skipping a stage because an artifact from a previous run already exists.

Baseline rationale:
Resume should be possible, but reuse must be explicit because stale artifacts can hide bad state.

Suggested policy:

- artifact existence alone is insufficient
- artifact must also pass stage validator
- if valid and user confirms reuse, state can mark stage as completed and continue
- if user does not confirm reuse, rerun the stage

### 12. Should the agent be able to resume interrupted runs?

Why this matters:
Long-running training and collection jobs need resumability.

Decision:
Yes, implement resumability.

Baseline rationale:
Yes. This is one of the strongest reasons to use LangGraph.

The agent state should be serializable and reloadable, with:

- current stage
- completed stages
- action history
- config history
- artifact map
- quality reports
- retry counters

### 13. What is the tool contract?

Why this matters:
If tools return inconsistent shapes, the agent layer becomes fragile.

Decision:
Use the proposed normalized tool contract.

Baseline rationale:
Every stage tool should be wrapped behind a normalized contract.

Recommended normalized `ActionResult` fields:

- `status`: `success | failed | partial`
- `stage`
- `artifacts`
- `metrics`
- `warnings`
- `raw_output`
- `quality_report`
- `state_patch`
- `error`

Even if the underlying repo tools currently return different structures, the agent should never consume them directly. Use adapters.

### 14. What should the agent log for inspection and debugging?

Why this matters:
If you cannot inspect decisions, you cannot trust the system.

Decision:
Use a minimal necessary trace, not full raw logging by default.

Baseline rationale:
Every step should log enough to debug decisions without making runs hard to inspect:

- candidate actions
- chosen action
- reason for choice
- tool input config
- quality report
- artifact paths
- retry count changes
- blocking issues

Full raw tool outputs can remain available behind a debug flag or artifact path reference.

### 15. Is the first version single-agent or multi-agent?

Why this matters:
This determines complexity and framework shape.

Decision:
Single-agent controller for v1.

Baseline rationale:
Single-agent controller first.

Do not start with a team of planner/judge/executor agents. The current repo does not have the signal quality or tool stability to justify that extra complexity.

### 16. Where should LLM reasoning actually be used?

Why this matters:
LLMs should be used only where they add value beyond rules.

Decision:
Use LLM reasoning only for the proposed bounded decision points.

Baseline rationale:
Use LLM reasoning for:

- interpreting ambiguous quality outcomes
- recommending bounded config changes
- ranking admissible next actions when several are plausible
- summarizing error clusters into actionable fixes

Do not use LLM reasoning for:

- file existence checks
- schema validation
- numeric threshold comparisons
- artifact dependency checks

### 17. What is the expected output of the agent itself?

Why this matters:
The agent should emit not just artifacts but a machine-readable decision trace.

Decision:
Use the proposed final agent outputs.

Baseline rationale:
At the end of each run, persist:

- final `PipelineState`
- `decision_history.jsonl`
- `quality_history.jsonl`
- `run_summary.json`

This becomes the basis for debugging and later training better policies.

### 18. What are the non-goals for v1?

Why this matters:
Clear non-goals prevent overbuilding.

Decision:
V1 is a constrained controller over a known graph.

Baseline rationale:
The first version should not try to do:

- open-ended tool discovery
- self-modifying prompts
- automatic code editing
- full multi-agent debate
- fully autonomous prompt engineering
- dynamic workflow graph generation

The first version should be a constrained controller over a known graph.

## Part 2: Vision for a Robust First Version Using LangGraph

This section describes the implementation vision that follows from Part 1. The aim is a reliable `full` mode agent that works with the current repo model and can evolve later.

## 2.1 Core Design Principle

Use LangGraph as a durable execution runtime for a constrained `full` pipeline controller, not as a substitute for domain logic.

That means:

- LangGraph manages state, routing, loops, and resumability
- repo tools execute pipeline stages
- custom validators determine quality
- a bounded decision policy decides the next step
- user confirmation is requested before reusing artifacts from previous runs

The architecture should look more like a stateful controller than a chat agent.

## 2.2 Recommended High-Level Architecture

### A. State Layer

Create a typed `PipelineState` model that contains:

- run metadata
- mode
- initial goal
- active config
- artifact registry
- stage statuses
- quality reports
- retry counters
- action history
- blocking issues
- final outcome
- resume confirmation status

Suggested modules:

- `core/agentic/state.py`
- `core/agentic/models.py`

### B. Tool Adapter Layer

Do not call repo tools directly from LangGraph nodes. Wrap each tool behind a stable adapter.

Suggested modules:

- `core/agentic/tool_adapters.py`

Responsibilities:

- convert from `PipelineState` to tool input config
- call the existing tool
- normalize outputs to `ActionResult`
- attach stage-local validation results

### C. Validation Layer

Implement deterministic validators for each stage.

Suggested modules:

- `core/agentic/validators/taxonomy.py`
- `core/agentic/validators/collection.py`
- `core/agentic/validators/sft.py`
- `core/agentic/validators/dataset.py`
- `core/agentic/validators/training.py`
- `core/agentic/validators/eval.py`
- `core/agentic/validators/reporting.py`

Each validator should return a standardized `QualityReport`.

### D. Decision Policy Layer

Implement a bounded policy that chooses among admissible actions.

Suggested modules:

- `core/agentic/policy.py`
- `core/agentic/action_space.py`

Responsibilities:

- enumerate candidate next actions based on state
- filter illegal actions
- prefer deterministic routing where obvious
- invoke LLM only when several legal actions remain or config adjustment requires interpretation
- enforce that v1 only uses the known `full` graph

### E. Resume Confirmation Layer

Add a small human checkpoint before skipping already-executed stages.

Suggested module:

- `core/agentic/resume.py`

Responsibilities:

- detect reusable artifacts from a previous run
- validate them before asking
- request confirmation to reuse or rerun
- record the answer in state

### F. LangSmith Observability Layer

Use LangSmith as the primary observability backend for the agent trajectory.

Suggested module:

- `core/agentic/observability.py`

Responsibilities:

- create one root trace for the full agent run
- create one child span per stage action
- attach action request, quality report, retry counts, artifacts, and final outcome
- require the LangSmith SDK as a project dependency
- require `LANGSMITH_API_KEY` for controller runs unless a test observer is injected
- do not maintain a separate custom trajectory renderer/exporter

Operational setup:

- install `langsmith`
- set `LANGSMITH_API_KEY`
- optionally set `LANGSMITH_PROJECT=horde-agent`

### G. LangGraph Runtime Layer

Implement the graph itself.

Suggested module:

- `agent/langgraph_orchestrator.py`

This should sit alongside the current workflow runner for the agentic `full` path.

## 2.3 Recommended Graph Shape for V1

The graph should be simple and explicit.

### Node 1: `initialize_run`

Responsibilities:

- create initial state
- resolve initial config
- enforce `full` mode for v1
- record goal
- inspect any pre-existing artifacts for resumability

### Node 2: `confirm_resume`

Responsibilities:

- identify previously completed stages with valid artifacts
- ask the user whether to reuse them or rerun them
- write confirmation decisions into state

This node should be skipped when there are no reusable prior artifacts.

### Node 3: `inspect_state`

Responsibilities:

- determine current pipeline position
- detect missing prerequisites
- compute admissible actions

### Node 4: `choose_next_action`

Responsibilities:

- choose the next legal action
- prefer deterministic routing
- invoke LLM only for bounded selection or bounded config adjustment

### Node 5: `execute_action`

Responsibilities:

- call the appropriate tool adapter
- capture raw outputs
- normalize result

### Node 6: `validate_action_result`

Responsibilities:

- run stage validator
- produce `QualityReport`
- update stage status

### Node 7: `apply_recovery_or_progress`

Responsibilities:

- if quality gate passed, advance
- if recoverable failure, increment retry counter and modify config if allowed
- if unrecoverable, mark blocking issue

### Node 8: `check_termination`

Responsibilities:

- stop on success
- stop on unrecoverable failure
- stop when retry or graph-step limits are exhausted
- otherwise loop back to `inspect_state`

### Terminal nodes

- `finish_success`
- `finish_failure`

This is enough for a solid first version.

## 2.4 Recommended Routing Strategy

Do not make routing fully agentic at first.

Use a layered routing model:

### Layer 1: deterministic routing

Examples:

- if taxonomy missing in `full`, you cannot train next
- if collection output missing, `build_sft_dataset` is illegal
- if training succeeded and eval missing, eval is the natural next step

### Layer 2: rule-based recovery

Examples:

- if sample count too low, rerun collection with broader settings
- if SFT invalid rate too high, rerun SFT with smaller batches or fallback parsing strategy
- if training unstable, reduce LR or effective batch size

### Layer 3: LLM-assisted tie-breaks

Examples:

- collection is mediocre and SFT is possible, but not great: should we retry collection first or continue to inspect downstream quality?
- training degraded slightly: should we spend one more retry?

This is where structured LLM decisions belong.

## 2.5 Suggested Typed Models

At minimum, define:

- `PipelineState`
- `StageStatus`
- `ArtifactRef`
- `ActionType`
- `ActionRequest`
- `ActionResult`
- `QualityReport`
- `TerminationDecision`
- `RecoveryPlan`

Important fields in `QualityReport`:

- `stage`
- `passed`
- `score`
- `blocking_issues`
- `warnings`
- `metrics`
- `suggested_adjustments`

Important fields in `PipelineState`:

- `goal`
- `mode`
- `run_dir`
- `config`
- `artifacts`
- `completed_stages`
- `current_stage`
- `quality_reports`
- `retry_counts`
- `decision_history`
- `resume_confirmations`
- `termination_reason`

## 2.6 LLM Usage in V1

Use one structured-output model for agent decisions.

The LLM should not be given full free-form authority. It should produce typed decisions such as:

- `NextActionDecision`
- `ConfigAdjustmentDecision`
- `RecoveryDecision`
- `ErrorInterpretationDecision`

These decisions should always be validated against policy constraints before execution.

In other words:

- LLM proposes
- policy checks legality
- runtime executes only legal proposals

## 2.7 Suggested File and Module Layout

Recommended additions:

- `agent/langgraph_orchestrator.py`
- `core/agentic/state.py`
- `core/agentic/models.py`
- `core/agentic/action_space.py`
- `core/agentic/policy.py`
- `core/agentic/tool_adapters.py`
- `core/agentic/llm_policy.py`
- `core/agentic/observability.py`
- `core/agentic/resume.py`
- `core/agentic/validators/__init__.py`
- `core/agentic/validators/taxonomy.py`
- `core/agentic/validators/collection.py`
- `core/agentic/validators/sft.py`
- `core/agentic/validators/dataset.py`
- `core/agentic/validators/training.py`
- `core/agentic/validators/eval.py`
- `core/agentic/validators/reporting.py`
- `tests/test_agentic_policy.py`
- `tests/test_agentic_validators.py`
- `tests/test_langgraph_orchestrator.py`

## 2.8 Recommended Implementation Order

To keep the project stable, implement in this order:

### Step 1: Normalize tool contracts

Wrap current tool outputs into `ActionResult`.

This is the most important prerequisite.

### Step 2: Build deterministic validators

Get hard gates working before adding any LLM policy logic.

### Step 3: Build `PipelineState`

Make state serialization clean and explicit.

### Step 4: Implement a deterministic policy

Before using the LLM, prove that:

- state transitions work
- retries work
- stop conditions work
- `full` graph ordering is enforced

### Step 5: Add resume confirmation

Validate previous artifacts and ask the user whether to reuse them before marking earlier stages complete.

### Step 6: Add LangSmith observability

Trace the root run and every stage action so the trajectory can be inspected in the LangSmith UI.

### Step 7: Port this logic into LangGraph nodes

Use LangGraph mainly for durability and loop control.

### Step 8: Add bounded LLM decision nodes

Only after deterministic behavior is stable.

### Step 9: Add resumability and run inspection tooling

This gives the system operational value.

## 2.9 What Makes This First Version Robust

The first version should be considered robust if:

- every action has a typed contract
- every stage has a validator
- retries are bounded and justified
- success and failure are both explicit end states
- runs can be resumed from serialized state
- stage reuse requires validation and user confirmation
- every decision is inspectable after the fact in LangSmith

Reliability in v1 should come from constraint, not intelligence.

## Part 3: Future Tweaks, Improvements, and Efficiency Gains

Once the first version works, these are the main axes to improve.

## 3.1 Better Quality Signals

The first validator set will be crude. Strong improvements include:

- language identification checks on collected text
- semantic diversity scoring for taxonomy and SFT examples
- source trust scoring by domain
- answer groundedness heuristics for generated examples
- better training health checks from richer log parsing
- downstream-task eval instead of only generic failure heuristics

These will make the agent's actions much more rational.

## 3.2 More Sophisticated Recovery Policies

Initial recovery logic will likely be simple threshold-driven retries. Later you can add:

- stage-specific recovery playbooks
- conditional fallback branches
- progressive widening of collection parameters
- dynamic reduction of batch sizes after repeated failures
- confidence-aware stop conditions

## 3.3 Better Config Search

Training adjustments can evolve from simple bounded tweaks to:

- offline-tuned recovery policies
- Bayesian search over a small safe config region
- multi-armed bandit selection for training presets
- stage-conditioned hyperparameter policies

For v1, keep this very small.

## 3.4 Selective Use of Specialist Subagents

If you later want more agentic depth, do not start with global multi-agent control. Instead add specialist sub-policies for bounded tasks:

- taxonomy quality reviewer
- collection quality reviewer
- SFT data quality reviewer
- training recovery advisor
- eval interpretation advisor

These should advise the main controller, not replace it.

## 3.5 Parallelism

Later efficiency gains can come from safe parallelism:

- parallel taxonomy sub-branches
- parallel collection batches
- parallel SFT annotation batches
- parallel trial training runs for cheap model candidates
- asynchronous artifact validation

Only add this after state and observability are reliable.

## 3.6 Better Human-in-the-Loop Controls

LangGraph can support pauses and resumes. Useful later additions:

- ask for approval before expensive retries
- pause on ambiguous quality outcomes
- allow manual stage override
- allow operator-supplied recovery instructions

This is especially useful when training costs increase.

## 3.7 Learned Policies From Decision Logs

If you persist decision history, later you can:

- analyze which retries are wasteful
- learn better threshold settings
- compare success rates across policies
- fine-tune prompts or rule ordering based on past runs

In other words, the first version should emit the data required to improve the second version.

## 3.8 Richer Artifact Graph

The current artifact model will likely be path-based. Later you can make it more expressive:

- versioned artifacts
- lineage links
- confidence scores
- provenance metadata
- checksum validation
- cache reuse across runs

This becomes increasingly important once resumability and branching grow.

## 3.9 Automatic Mode Selection

A later agent version may choose among:

- full collection pipeline
- workflow mode from existing dataset
- fast debug mode
- train-only mode
- eval-only mode

For v1, mode is fixed to `full`. Later versions can make mode selection explicit or agent-assisted.

## 3.10 Smarter Economic Decisions

A stronger agent should eventually decide not only what is possible but what is worth doing.

Useful future additions:

- expected-value estimation for another retry
- budget-aware stop decisions
- stage-specific utility scoring
- tradeoff between more data and more training

This is where the system becomes strategically efficient rather than just operationally correct.

## 3.11 UI and Observability Improvements

Once the backend is stable, expose LangSmith trace links and agent state in UI:

- current state
- last action
- next candidate actions
- quality dashboards
- retry counters
- failure reasons
- LangSmith run URL

This would turn the current mock-style UI into a meaningful operations interface.

## 3.12 Prompt and Policy Upgrades

Later, improve decision quality with:

- better structured system prompts
- separate prompts for recovery versus planning
- rubric-driven LLM evaluations
- prompt versions tracked in state

But prompt work should never substitute for weak state design or missing validators.

## Final Recommendation

The best implementation path for this repo is:

1. Keep the existing tools and pipeline stages.
2. Build a normalized contract layer around them.
3. Define explicit quality gates and recovery rules.
4. Implement a single-controller agent with typed state for the `full` graph.
5. Use LangGraph to run that controller durably and inspectably.
6. Add LLM-based bounded decisions only after deterministic behavior is working.

If you follow that order, you will get a real agentic system rather than a workflow with agent-flavored text around it.
