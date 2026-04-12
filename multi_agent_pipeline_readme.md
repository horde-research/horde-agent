# Multi-Agent Pipeline for Low-Resource Language Data Collection

## Goal
Build a multi-agent system to collect culturally-specific data for low-resource languages and improve model performance.

## Architecture
[Taxonomy Agent] → [Query Planning Agent] → [Crawler] → [Cleaning] → [Candidate Generation Agent] → [Judge Agent] → [Dataset Builder] → [Train/Eval]

## Agents
- Taxonomy Agent: builds cultural topic space
- Query Planning Agent: generates search strategies
- Candidate Generation Agent: converts text → training data
- Judge Agent: filters and validates data

## Non-Agent Components
- Crawler
- Cleaning (langid, dedup)
- Dataset builder
- Training pipeline

## Development Plan
Phase 0: baseline pipeline
Phase 1: taxonomy agent
Phase 2: query planning
Phase 3: candidate generation
Phase 4: judge

## Hypotheses
H1: taxonomy improves coverage
H2: query planning improves precision
H3: candidate generation improves data quality
H4: judge reduces hallucinations

## Rules
- no agent should generate + validate same data
- deterministic filters between agents
- measure each stage

## Success Criteria
- higher cultural specificity
- stable acceptance rate
- improved eval metrics
