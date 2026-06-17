"""Language-agnostic visual taxonomy for image data collection.

The taxonomy uses stable English/ASCII IDs so downstream training contracts do
not depend on a particular language. Culture-specific adaptation happens only
in generated search queries through culture aliases from ``infer_culture_profile``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

IMAGE_TAXONOMY_SCHEMA_VERSION = "image_taxonomy_v1"

IMAGE_TAXONOMY: List[Dict[str, Any]] = [
    {
        "domain_id": "clothing_textiles_adornment",
        "domain_label": "Clothing, Textiles, and Adornment",
        "description": "Garments, textiles, patterns, jewelry, headwear, ceremonial dress, and everyday fashion.",
        "visual_skills": ["attribute_recognition", "material_texture", "pattern_description", "human_activity"],
        "subdomains": [
            {
                "subdomain_id": "traditional_clothing",
                "subdomain_label": "Traditional Clothing",
                "description": "Culturally distinctive garments worn in ceremonial, historical, or everyday contexts.",
                "search_terms": ["traditional clothing", "national costume", "ceremonial dress"],
                "query_intents": ["person_wearing_item", "ceremonial_context", "object_closeup"],
            },
            {
                "subdomain_id": "contemporary_ethnic_fashion",
                "subdomain_label": "Contemporary Ethnic Fashion",
                "description": "Modern clothing that visibly adapts local motifs, textiles, silhouettes, or accessories.",
                "search_terms": ["modern ethnic fashion", "contemporary national clothing", "urban fashion"],
                "query_intents": ["modern_everyday_context", "person_wearing_item", "group_scene"],
            },
            {
                "subdomain_id": "textiles_patterns",
                "subdomain_label": "Textiles and Patterns",
                "description": "Fabric, embroidery, woven patterns, ornaments, colors, and repeated motifs.",
                "search_terms": ["traditional textile patterns", "ornament close up", "embroidery details"],
                "query_intents": ["object_closeup", "material_texture", "pattern_detail"],
            },
        ],
    },
    {
        "domain_id": "food_drink_table_culture",
        "domain_label": "Food, Drink, and Table Culture",
        "description": "Prepared dishes, ingredients, table settings, serving practices, and food-related social scenes.",
        "visual_skills": ["object_recognition", "counting", "spatial_layout", "material_texture"],
        "subdomains": [
            {
                "subdomain_id": "traditional_food",
                "subdomain_label": "Traditional Food",
                "description": "Recognizable traditional dishes, ingredients, preparation, and serving styles.",
                "search_terms": ["traditional food", "national dish", "traditional meal"],
                "query_intents": ["object_closeup", "table_scene", "preparation_process"],
            },
            {
                "subdomain_id": "table_setting_hospitality",
                "subdomain_label": "Table Setting and Hospitality",
                "description": "Meal layouts, guest settings, serving vessels, table textiles, and hospitality scenes.",
                "search_terms": ["traditional table setting", "hospitality meal", "family feast"],
                "query_intents": ["table_scene", "group_scene", "spatial_layout"],
            },
        ],
    },
    {
        "domain_id": "architecture_living_spaces",
        "domain_label": "Architecture and Living Spaces",
        "description": "Homes, interiors, monuments, vernacular architecture, urban forms, and public buildings.",
        "visual_skills": ["scene_description", "spatial_relationship", "material_texture", "ocr"],
        "subdomains": [
            {
                "subdomain_id": "traditional_dwellings",
                "subdomain_label": "Traditional Dwellings",
                "description": "Locally specific historical or vernacular homes, shelters, interiors, and furnishings.",
                "search_terms": ["traditional dwelling", "traditional home interior", "vernacular architecture"],
                "query_intents": ["interior_scene", "exterior_scene", "material_texture"],
            },
            {
                "subdomain_id": "urban_landmarks_public_buildings",
                "subdomain_label": "Urban Landmarks and Public Buildings",
                "description": "Recognizable public architecture, monuments, city squares, and civic buildings.",
                "search_terms": ["city landmark", "public building", "monument architecture"],
                "query_intents": ["exterior_scene", "public_space", "ocr_signage"],
            },
        ],
    },
    {
        "domain_id": "landscape_ecology_nature",
        "domain_label": "Landscape, Ecology, and Nature",
        "description": "Landforms, climate, protected areas, natural resources, and human-environment interaction.",
        "visual_skills": ["scene_description", "fine_grained_classification", "spatial_layout", "reasoning"],
        "subdomains": [
            {
                "subdomain_id": "distinctive_landscapes",
                "subdomain_label": "Distinctive Landscapes",
                "description": "Geographically distinctive landforms, natural scenes, seasons, and regional environments.",
                "search_terms": ["distinctive landscape", "national park landscape", "regional nature"],
                "query_intents": ["wide_scene", "seasonal_context", "geographic_feature"],
            },
            {
                "subdomain_id": "environmental_activity",
                "subdomain_label": "Environmental Activity",
                "description": "Visible conservation, clean-up, sustainability, agriculture, water, or energy activities.",
                "search_terms": ["environmental initiative", "renewable energy", "conservation activity"],
                "query_intents": ["human_activity", "infrastructure_scene", "public_activity"],
            },
        ],
    },
    {
        "domain_id": "animals_plants_local_environment",
        "domain_label": "Animals, Plants, and Local Environment",
        "description": "Locally important animals, plants, habitats, pastoral life, and human-animal relations.",
        "visual_skills": ["fine_grained_classification", "counting", "attribute_recognition", "reasoning"],
        "subdomains": [
            {
                "subdomain_id": "local_animals",
                "subdomain_label": "Local Animals",
                "description": "Wildlife, working animals, pastoral animals, and culturally salient species.",
                "search_terms": ["local wildlife", "traditional animal husbandry", "native animal"],
                "query_intents": ["animal_closeup", "habitat_scene", "human_animal_activity"],
            },
            {
                "subdomain_id": "local_plants_habitats",
                "subdomain_label": "Local Plants and Habitats",
                "description": "Characteristic plants, habitats, flowers, forests, grasslands, and seasonal vegetation.",
                "search_terms": ["local plants", "native flowers", "natural habitat"],
                "query_intents": ["plant_closeup", "habitat_scene", "seasonal_context"],
            },
        ],
    },
    {
        "domain_id": "daily_life_work_transport",
        "domain_label": "Daily Life, Work, and Transport",
        "description": "Everyday activities, markets, work scenes, rural and urban transport, and social spaces.",
        "visual_skills": ["human_activity", "object_recognition", "spatial_relationship", "counting"],
        "subdomains": [
            {
                "subdomain_id": "markets_workplaces",
                "subdomain_label": "Markets and Workplaces",
                "description": "Street markets, workshops, professional settings, agriculture, and visible labor practices.",
                "search_terms": ["local market", "traditional workplace", "daily work"],
                "query_intents": ["human_activity", "object_counting", "public_space"],
            },
            {
                "subdomain_id": "transport_mobility",
                "subdomain_label": "Transport and Mobility",
                "description": "Vehicles, roads, public transport, rural mobility, and movement through public space.",
                "search_terms": ["public transport", "rural transport", "street traffic"],
                "query_intents": ["vehicle_scene", "public_space", "spatial_layout"],
            },
        ],
    },
    {
        "domain_id": "rituals_festivals_performance",
        "domain_label": "Rituals, Festivals, and Performance",
        "description": "Ceremonies, festivals, public celebrations, music, dance, sports, and staged performance.",
        "visual_skills": ["human_activity", "counting", "spatial_layout", "attribute_recognition"],
        "subdomains": [
            {
                "subdomain_id": "festivals_rituals",
                "subdomain_label": "Festivals and Rituals",
                "description": "Public or private celebrations, rituals, processions, ceremonies, and seasonal events.",
                "search_terms": ["traditional festival", "cultural ceremony", "public celebration"],
                "query_intents": ["group_scene", "ceremonial_context", "public_activity"],
            },
            {
                "subdomain_id": "music_dance_sport",
                "subdomain_label": "Music, Dance, and Sport",
                "description": "Performers, instruments, dance movement, traditional games, and visible sport activities.",
                "search_terms": ["traditional music performance", "folk dance", "traditional sport"],
                "query_intents": ["performance_scene", "object_closeup", "human_activity"],
            },
        ],
    },
    {
        "domain_id": "arts_crafts_objects_instruments",
        "domain_label": "Arts, Crafts, Objects, and Instruments",
        "description": "Handmade objects, tools, instruments, artworks, decorative items, and material culture.",
        "visual_skills": ["object_recognition", "material_texture", "fine_grained_classification", "attribute_recognition"],
        "subdomains": [
            {
                "subdomain_id": "craft_objects",
                "subdomain_label": "Craft Objects",
                "description": "Handicrafts, household objects, tools, decorative objects, and production processes.",
                "search_terms": ["traditional craft object", "handicraft close up", "artisan workshop"],
                "query_intents": ["object_closeup", "material_texture", "production_process"],
            },
            {
                "subdomain_id": "musical_instruments",
                "subdomain_label": "Musical Instruments",
                "description": "Instruments, performance contexts, construction details, and playing posture.",
                "search_terms": ["traditional musical instrument", "instrument close up", "musician playing"],
                "query_intents": ["object_closeup", "performance_scene", "human_activity"],
            },
        ],
    },
    {
        "domain_id": "writing_signage_symbols",
        "domain_label": "Writing, Signage, and Symbols",
        "description": "Visible writing systems, signs, labels, public text, logos, maps, emblems, and symbolic marks.",
        "visual_skills": ["ocr", "symbol_recognition", "scene_description", "spatial_relationship"],
        "subdomains": [
            {
                "subdomain_id": "public_signage",
                "subdomain_label": "Public Signage",
                "description": "Street signs, shop signs, public notices, transport signs, and institutional signage.",
                "search_terms": ["street signs", "public signage", "shop signs"],
                "query_intents": ["ocr_signage", "public_space", "text_closeup"],
            },
            {
                "subdomain_id": "symbols_emblems_maps",
                "subdomain_label": "Symbols, Emblems, and Maps",
                "description": "Flags, emblems, visual symbols, maps, stamps, and official or informal iconography.",
                "search_terms": ["national symbols", "emblem", "map sign"],
                "query_intents": ["symbol_closeup", "object_closeup", "ocr_signage"],
            },
        ],
    },
]

_INTENT_TEMPLATES = {
    "animal_closeup": "{alias} {term} animal close up photo",
    "ceremonial_context": "{alias} {term} ceremony photo",
    "exterior_scene": "{alias} {term} exterior photo",
    "geographic_feature": "{alias} {term} geographic feature photo",
    "group_scene": "{alias} {term} group scene photo",
    "habitat_scene": "{alias} {term} habitat photo",
    "human_activity": "{alias} {term} people activity photo",
    "human_animal_activity": "{alias} {term} people with animals photo",
    "infrastructure_scene": "{alias} {term} infrastructure photo",
    "interior_scene": "{alias} {term} interior photo",
    "material_texture": "{alias} {term} texture close up photo",
    "modern_everyday_context": "{alias} {term} modern everyday photo",
    "object_closeup": "{alias} {term} close up photo",
    "object_counting": "{alias} {term} multiple objects photo",
    "ocr_signage": "{alias} {term} readable text signage photo",
    "pattern_detail": "{alias} {term} pattern detail photo",
    "performance_scene": "{alias} {term} performance photo",
    "person_wearing_item": "{alias} {term} person wearing photo",
    "plant_closeup": "{alias} {term} plant close up photo",
    "preparation_process": "{alias} {term} preparation process photo",
    "production_process": "{alias} {term} making process photo",
    "public_activity": "{alias} {term} public activity photo",
    "public_space": "{alias} {term} public space photo",
    "seasonal_context": "{alias} {term} seasonal photo",
    "spatial_layout": "{alias} {term} spatial layout photo",
    "symbol_closeup": "{alias} {term} symbol close up photo",
    "table_scene": "{alias} {term} table scene photo",
    "text_closeup": "{alias} {term} text close up photo",
    "vehicle_scene": "{alias} {term} vehicles photo",
    "wide_scene": "{alias} {term} wide landscape photo",
}


def build_image_taxonomy(
    country_or_culture: str,
    culture_profile: Dict[str, Any] | None = None,
    *,
    queries_per_slot: int = 4,
    max_slots: int | None = None,
) -> Dict[str, Any]:
    domains = deepcopy(IMAGE_TAXONOMY)
    aliases = _culture_aliases(country_or_culture, culture_profile or {})
    slots: List[Dict[str, Any]] = []

    for domain in domains:
        for subdomain in domain["subdomains"]:
            slot = {
                "slot_id": f"{domain['domain_id']}__{subdomain['subdomain_id']}",
                "domain_id": domain["domain_id"],
                "domain_label": domain["domain_label"],
                "subdomain_id": subdomain["subdomain_id"],
                "subdomain_label": subdomain["subdomain_label"],
                "description": subdomain["description"],
                "visual_skills": list(domain["visual_skills"]),
                "query_intents": list(subdomain["query_intents"]),
                "queries": _build_slot_queries(
                    aliases=aliases,
                    search_terms=subdomain["search_terms"],
                    query_intents=subdomain["query_intents"],
                    queries_per_slot=max(1, int(queries_per_slot)),
                ),
            }
            slots.append(slot)
            if max_slots and len(slots) >= max_slots:
                return _taxonomy_payload(country_or_culture, domains, slots)

    return _taxonomy_payload(country_or_culture, domains, slots)


def flatten_image_query_specs(image_taxonomy: Dict[str, Any]) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    for slot in image_taxonomy.get("slots", []):
        for query_entry in slot.get("queries", []):
            if isinstance(query_entry, str):
                query = query_entry
                query_intent = ""
            else:
                query = str(query_entry.get("query") or "")
                query_intent = str(query_entry.get("query_intent") or "")
            query = query.strip()
            if not query:
                continue
            specs.append(
                {
                    "query": query,
                    "query_intent": query_intent,
                    "slot_id": str(slot.get("slot_id") or ""),
                    "domain_id": str(slot.get("domain_id") or ""),
                    "domain_label": str(slot.get("domain_label") or ""),
                    "subdomain_id": str(slot.get("subdomain_id") or ""),
                    "subdomain_label": str(slot.get("subdomain_label") or ""),
                    "visual_skills": ",".join(str(skill) for skill in slot.get("visual_skills", [])),
                }
            )
    return specs


def _taxonomy_payload(country_or_culture: str, domains: List[Dict[str, Any]], slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": IMAGE_TAXONOMY_SCHEMA_VERSION,
        "country_or_culture": country_or_culture,
        "domains": domains,
        "slots": slots,
        "quality": {
            "passed": bool(slots),
            "num_domains": len(domains),
            "num_slots": len(slots),
            "num_queries": sum(len(slot.get("queries", [])) for slot in slots),
        },
    }


def _build_slot_queries(
    *,
    aliases: List[str],
    search_terms: List[str],
    query_intents: List[str],
    queries_per_slot: int,
) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []
    seen: set[str] = set()

    for alias in aliases:
        for intent in query_intents:
            for term in search_terms:
                template = _INTENT_TEMPLATES.get(intent, "{alias} {term} photo")
                query = " ".join(template.format(alias=alias, term=term).split())
                if query.lower() in seen:
                    continue
                seen.add(query.lower())
                queries.append({"query": query, "query_intent": intent})
                if len(queries) >= queries_per_slot:
                    return queries

    return queries


def _culture_aliases(country_or_culture: str, culture_profile: Dict[str, Any]) -> List[str]:
    aliases = [country_or_culture]
    aliases.extend(str(alias) for alias in culture_profile.get("common_aliases", []) if alias)
    return list(_dedupe(alias.strip() for alias in aliases if alias and alias.strip()))


def _dedupe(values: Iterable[str]) -> Iterable[str]:
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        yield value
