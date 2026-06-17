from __future__ import annotations

from tools.generate_taxonomy.image_taxonomy import build_image_taxonomy, flatten_image_query_specs
from tools.generate_taxonomy.quality import infer_culture_profile


def test_image_taxonomy_uses_stable_language_agnostic_ids() -> None:
    taxonomy = build_image_taxonomy(
        "Kazakhstan",
        infer_culture_profile("Kazakhstan"),
        queries_per_slot=2,
        max_slots=4,
    )

    assert taxonomy["schema_version"] == "image_taxonomy_v1"
    assert len(taxonomy["slots"]) == 4
    assert taxonomy["quality"]["passed"] is True

    for slot in taxonomy["slots"]:
        assert slot["domain_id"].isascii()
        assert slot["subdomain_id"].isascii()
        assert " " not in slot["domain_id"]
        assert " " not in slot["subdomain_id"]
        assert slot["visual_skills"]
        assert len(slot["queries"]) == 2


def test_image_taxonomy_query_specs_preserve_slot_metadata() -> None:
    taxonomy = build_image_taxonomy(
        "Kazakhstan",
        infer_culture_profile("Kazakhstan"),
        queries_per_slot=2,
        max_slots=1,
    )

    specs = flatten_image_query_specs(taxonomy)

    assert len(specs) == 2
    assert specs[0]["query"]
    assert specs[0]["slot_id"] == taxonomy["slots"][0]["slot_id"]
    assert specs[0]["domain_id"] == taxonomy["slots"][0]["domain_id"]
    assert specs[0]["subdomain_id"] == taxonomy["slots"][0]["subdomain_id"]
    assert specs[0]["query_intent"]
