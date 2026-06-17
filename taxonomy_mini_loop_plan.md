# Taxonomy Mini-Loop Plan

This document describes the first high-impact place to add real agentic behavior inside the current workflow: `GenerateTaxonomyTool`.

The current full pipeline depends heavily on taxonomy quality. If categories, subcategories, or search queries are weak, data collection becomes weak, SFT examples become weak, and training/evaluation quality is downstream noise. The goal of this mini-loop is to make taxonomy generation self-checking and selectively self-repairing before collection starts.

## Current Flow

Current implementation:

1. `CategoryAgent.extract_categories(country_or_culture)`
2. `SubcategoryAgent.generate_for_categories(categories, country_or_culture)`
3. `QueryAgent.generate_for_subcategories(categories, category_subcategories, country_or_culture)`
4. Return taxonomy output directly

Relevant files:

- `tools/generate_taxonomy/tool.py`
- `tools/generate_taxonomy/agents/category_agent.py`
- `tools/generate_taxonomy/agents/subcategory_agent.py`
- `tools/generate_taxonomy/agents/keyword_agent.py`

Current limitation:

- no quality gate before moving from categories to subcategories
- no repair for missing/overlapping categories
- no targeted retry for weak subcategory groups
- no validation of query count, query diversity, or multilingual coverage before collection

## Target Flow

Recommended mini-loop:

1. Infer culture profile
2. Generate categories
3. Validate and review categories
4. Refine categories if quality is insufficient
5. Generate subcategories
6. Validate and repair only weak subcategory groups
7. Generate search queries
8. Validate and repair only weak query groups
9. Return taxonomy plus quality metadata

The important design choice is selective repair. Do not regenerate the whole taxonomy unless the top-level category set is structurally bad. Most failures should be repaired at the smallest affected unit.

## Step 1: Culture Profile

Add a lightweight profile before category generation.

Profile fields:

- `country_or_culture`
- `native_languages`
- `native_scripts`
- `common_aliases`
- `important_regions`
- `traditional_domains`
- `modern_domains`
- `query_language_mix`

Purpose:

- guide category generation with culture-specific context
- improve native-language query generation
- avoid generic categories that could apply to any country
- make quality checks more concrete

For Kazakhstan, the profile should push the model toward Kazakh Cyrillic, Russian, English, regional variation, traditional nomadic culture, modern urban culture, cuisine, crafts, music, sports, religion, and contemporary social trends.

## Step 2: Category Generation

Keep the current `CategoryAgent.extract_categories(...)` as the first generator.

Expected output:

- 8-15 categories
- each category has `name`
- each category has meaningful `description`
- category names are distinct
- category set covers both traditional and contemporary culture

## Step 3: Category Quality Gate

Deterministic checks:

- category count between configured min/max
- no empty names
- no empty descriptions
- duplicate name rate below threshold
- names are not too generic
- required cultural dimensions have at least rough coverage

Suggested required dimensions:

- cuisine and food culture
- music, arts, literature, oral tradition
- customs, rituals, celebrations
- language and communication
- clothing, crafts, visual culture
- architecture and living spaces
- family, social structures, daily life
- geography, regional variation, environment
- economy, work, professional life
- modern trends, youth culture, technology

LLM reviewer checks:

- missing important domains
- overlapping categories
- categories too broad to be actionable
- categories too narrow for top-level taxonomy
- categories not specific to the target culture
- suggested merges/additions/removals

Reviewer output should be structured:

```json
{
  "passed": false,
  "score": 0.72,
  "missing_domains": ["regional variation"],
  "overlapping_categories": [["arts", "crafts"]],
  "too_generic": ["daily life"],
  "recommended_additions": ["regional identities and landscape"],
  "recommended_merges": [],
  "repair_instruction": "Add regional/environmental category and make daily life more Kazakhstan-specific."
}
```

## Step 4: Category Refinement

If category quality fails, call a refinement prompt.

Inputs:

- original categories
- culture profile
- deterministic gate failures
- reviewer feedback

Output:

- full revised category list, not a patch

Retry rule:

- max 2 category refinement attempts
- stop with blocking issue if still failing

Reason:

Top-level category quality controls every downstream branch. It is worth spending one or two extra LLM calls here.

## Step 5: Subcategory Generation

Keep current batched `SubcategoryAgent.generate_for_categories(...)`.

Expected output per category:

- 4-10 subcategories
- each has name and description
- each fits parent category
- subcategories are distinct and actionable
- includes traditional and contemporary coverage where relevant

## Step 6: Subcategory Quality Gate And Repair

Validate per category, not globally.

Deterministic checks:

- subcategory count per category within min/max
- no empty names
- no empty descriptions
- duplicate rate below threshold
- all parent categories have at least one valid subcategory

LLM reviewer checks:

- subcategories fit parent category
- important parent-category aspects are missing
- subcategories overlap heavily
- subcategories are too generic

Repair policy:

- only repair failed categories
- do not regenerate valid category branches
- max 2 repair attempts per failed category

Example:

If `Cuisine` is good but `Language & Communication` has only one generic subcategory, only rerun `Language & Communication`.

## Step 7: Search Query Generation

Keep current `QueryAgent.generate_for_subcategories(...)`.

Expected output per subcategory:

- 8-12 queries
- query length roughly 5-15 words
- every query includes target culture/country or a strong alias
- mix of English and native-language queries
- includes native script where appropriate
- avoids duplicate or near-duplicate queries
- covers different search intents

Search intents:

- overview/history
- traditions/practices
- examples/notable cases
- regional variation
- modern/current usage
- academic/reference style

## Step 8: Query Quality Gate And Repair

This is likely the highest ROI repair point before collection.

Deterministic checks:

- query count per subcategory
- duplicate rate
- minimum query length
- country/culture mention or alias present
- English/native-language balance
- native script present when expected
- no empty strings

LLM reviewer checks:

- queries are too broad
- queries are not searchable
- queries do not match subcategory
- queries over-focus on one aspect
- queries miss obvious culture-specific terms

Repair policy:

- only repair failed query groups
- pass the failed queries, subcategory, parent category, culture profile, and gate failures
- max 2 repair attempts per query group

Optional later enhancement:

- run a cheap Serper probe for a small query sample
- repair queries that return empty/irrelevant results before full collection

## Output Contract

`GenerateTaxonomyTool.execute(...)` should eventually return:

```json
{
  "categories": [],
  "category_subcategories": {},
  "category_subcategory_queries": {},
  "taxonomy_quality": {
    "passed": true,
    "score": 0.91,
    "category_report": {},
    "subcategory_report": {},
    "query_report": {},
    "repair_attempts": []
  }
}
```

The current top-level keys should remain backward compatible.

## Agentic Behavior Introduced

This mini-loop adds real agentic behavior because the tool will:

- inspect intermediate outputs
- decide whether output quality is sufficient
- identify the smallest failing unit
- retry or refine only that unit
- stop early on unrecoverable taxonomy failure
- return quality metadata for the outer `full_agentic` controller

This is stronger than only choosing which pipeline tool to call next.

## Implementation Order

Recommended order:

1. Add deterministic taxonomy validators.
2. Add tests for category, subcategory, and query quality gates.
3. Add structured reviewer schemas.
4. Add category refinement loop.
5. Add targeted subcategory repair loop.
6. Add targeted query repair loop.
7. Add taxonomy quality metadata to tool output.
8. Update `full_agentic` taxonomy adapter to consume `taxonomy_quality`.

## First Version Scope

For the first implementation, keep it constrained:

- no web search feedback loop yet
- no dynamic category count optimization
- no new external tools
- max 2 repair attempts per layer
- deterministic gates decide hard failures
- LLM reviewer provides repair guidance, not final authority

The first version should improve taxonomy quality while remaining predictable and testable.
