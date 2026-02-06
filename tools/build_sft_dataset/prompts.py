from typing import Optional


def build_image_prompt(
    target_language: str = "English",
    topic_hint: Optional[str] = None,
) -> str:
    topic_line = ""
    if topic_hint:
        topic_line = (
            f"\nContext: This image is from a collection related to '{topic_hint}'. "
            "Reference this ONLY if the visual content clearly aligns with this topic. "
            "If the image does not match the topic, ignore it entirely — describe only "
            "what you actually see."
        )

    return f"""You are a world-class AI training-data engineer. Your sole purpose is to
produce supervised fine-tuning (SFT) examples that will teach a vision-language
model to perceive, reason about, and describe visual content with expert-level
accuracy.{topic_line}

Target language for ALL generated text: {target_language}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES  (violating any of these makes the example harmful for training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ZERO HALLUCINATION — mention ONLY objects, attributes, actions, and spatial
   relationships that are directly, unambiguously visible in the image.
2. ZERO VAGUENESS — never say "some", "various", "a few", "appears to be",
   "it seems". Be concrete: exact counts, specific colors, clear positions.
3. ZERO FILLER — never start with "The image shows", "This is a picture of",
   "In this image we can see". Start directly with the content.
4. EVERY question must be answerable by viewing the image alone — no world
   knowledge, no assumptions about what is outside the frame.
5. EVERY answer must be faithful to visible evidence — if uncertain, say so.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROTOCOL  (follow in order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — ANALYZE (populate the `info` block first)
  Examine the image systematically before writing any training content:
  • Quality: Sharp/clear → High | Acceptable → Medium | Blurry/low-res → Low/Blurry
  • Primary subject: dominant element in the frame
  • Object count: number of distinct, prominent objects
  • Human presence: count people if any
  • Text: any readable text? type?
  • Task suitability: counting? reasoning? multi-step instructions?

STEP 2 — GENERATE (use your Step 1 analysis to guide every field)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD-BY-FIELD REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. caption  (Visual Comprehension)
   Length: 4-8 sentences, 50-120 words.
   Structure: subject → setting/background → colors & materials → actions →
              spatial layout → notable details.
   REQUIRED: at least 2 spatial relationships (e.g. "to the left of",
             "in the foreground", "mounted above").
   REQUIRED: at least 2 specific color references (e.g. "deep navy", not "dark").
   FORBIDDEN: opinion words (beautiful, nice, interesting, impressive).
   FORBIDDEN: "The image shows…", "There is…", "We can see…".

   GOOD: "A red double-decker bus travels down a narrow cobblestone street
   lined with three-story brownstone buildings. Three pedestrians walk along
   the left sidewalk; the nearest carries a bright blue umbrella. A green
   metal street sign reading 'Baker St' is mounted on the corner building
   at roughly second-floor height. Overcast grey clouds fill the sky. In the
   far background, a white delivery van is parked beside the kerb."

   BAD: "A nice street scene with a bus and some people walking. It looks
   like a European city on a cloudy day."

2. vqa  (Visual Question Answering — exactly 5 pairs)
   Each pair must test a DISTINCT visual skill:

   a) OBJECT / ATTRIBUTE IDENTIFICATION
      Ask about a specific object's color, type, brand, shape, or material.
      Q must name or locate the object unambiguously.
      GOOD Q: "What color is the double-decker bus on the cobblestone street?"
      BAD  Q: "What color is the bus?"  (which bus?)

   b) COUNTING / QUANTIFICATION
      Ask for an exact count of clearly distinguishable items.
      GOOD Q: "How many pedestrians are visible on the left sidewalk?"
      BAD  Q: "How many people?"  (unclear scope)

   c) SPATIAL RELATIONSHIP / POSITIONING
      Ask where one object is relative to another.
      GOOD Q: "Where is the green street sign positioned relative to the
               corner building?"
      BAD  Q: "Where is the sign?"

   d) ACTION / ACTIVITY RECOGNITION
      Ask what a person, animal, or vehicle is doing.
      GOOD Q: "What is the nearest pedestrian on the left sidewalk carrying?"
      BAD  Q: "What are people doing?"

   e) DETAIL / TEXTURE / MATERIAL
      Ask about surface, material, texture, pattern, or fine-grained visual
      properties.
      GOOD Q: "What type of surface covers the street the bus is driving on?"
      BAD  Q: "Describe the road."

   ALL answers: 1-3 sentences. Specific, grounded, no hedging.

3. ocr  (Text Recognition)
   If `info.text_properties.is_suitable_for_ocr` is TRUE:
     • instruction: "Read and transcribe all text visible on [specific location]."
     • answer: exact text as printed (preserve capitalization, punctuation).
   If FALSE: set both fields to null.

4. reason  (Visual Reasoning)
   If `info.task_suitability.is_suitable_for_reasoning` is TRUE:
     • Ask a question that requires combining 2+ visual cues to reach a
       conclusion.
     • The answer MUST cite each piece of visual evidence explicitly, then
       state the conclusion.
     GOOD Q: "Based on the overcast sky, the pedestrian's umbrella, and
              the dark, wet-looking cobblestones, what weather conditions
              can be inferred?"
     GOOD A: "Three visual cues point to recent or ongoing rain: (1) the
              uniformly grey overcast sky, (2) the nearest pedestrian
              carrying an open blue umbrella, and (3) the cobblestone
              surface that appears darker and glossier than dry stone
              typically looks. Together these strongly suggest it is
              raining or has very recently rained."
   If FALSE:
     • Ask a straightforward identification question.
     • In the answer, note that the image lacks sufficient contextual
       elements for deeper inference.

5. instruct_follow  (Task Execution)
   If `info.task_suitability.is_suitable_for_multi_step_instruction` is TRUE:
     • Create a 2-3 step composite instruction that requires the model to:
       first identify, then describe, then compare or classify.
     GOOD: "First, identify all vehicles visible in the scene. Then, for
            each vehicle, describe its color and type. Finally, state which
            vehicle is closest to the camera."
   If FALSE:
     • Create a focused single-step instruction.
     GOOD: "List every distinct color you can identify in this image."

   Answer must follow the instruction step by step.

6. info  (Structured Metadata — populate FIRST, before other fields)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond with a SINGLE valid JSON object — no markdown fences, no commentary.
Every text field must be in {target_language}.

{{
  "info": {{
    "content_properties": {{
      "quality": "High" | "Medium" | "Low/Blurry",
      "primary_subject": "Person/People" | "Animal(s)" | "Vehicle(s)" | "Food" | "Object(s)" | "Scenery/Architecture" | "Other",
      "object_count": <integer>,
      "human_presence": "None" | "Single Person" | "Small Group (2-5)" | "Crowd (>5)",
      "face_visibility": "Clear Faces" | "Obscured/No Faces" | "Not Applicable"
    }},
    "text_properties": {{
      "contains_text": <boolean>,
      "is_suitable_for_ocr": <boolean>,
      "text_type": "Printed" | "Handwritten" | "Signage/Logo" | "Not Applicable"
    }},
    "task_suitability": {{
      "is_suitable_for_counting": <boolean>,
      "is_suitable_for_reasoning": <boolean>,
      "is_suitable_for_multi_step_instruction": <boolean>
    }}
  }},
  "caption": {{"text": "..."}},
  "vqa": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ],
  "ocr": {{"instruction": "..." | null, "answer": "..." | null}},
  "reason": {{"instruction": "...", "answer": "..."}},
  "instruct_follow": {{"instruction": "...", "answer": "..."}}
}}"""


def build_text_prompt(target_language: str = "English") -> str:
    return f"""You are a world-class AI training-data engineer. You will receive a
SOURCE TEXT that contains factual information about a topic, culture, or domain.

YOUR TASK: use the source text as REFERENCE MATERIAL to produce STANDALONE
supervised fine-tuning (SFT) training examples. Each training example is a
(question, answer) pair that will be used DIRECTLY to fine-tune a language
model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL PARADIGM — KNOWLEDGE DISTILLATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The source text will **NOT** appear in the training data. The training pairs
you generate must be COMPLETELY SELF-CONTAINED:

  • Questions must sound like a real person asking — natural, curious, specific.
  • Answers must contain ALL relevant knowledge extracted from the source.
  • A reader must fully understand both the Q and A WITHOUT ever seeing the
    source text.

Think of it this way: you are DISTILLING knowledge from the source INTO
training pairs that will TEACH a model this knowledge permanently.

Target language for ALL generated text: {target_language}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GROUNDED — every fact in an answer must originate from the source text.
   No external knowledge, no speculation, no hallucination.
2. STANDALONE — never say "the text says", "according to the passage",
   "as mentioned above". The source text does not exist in training.
3. KNOWLEDGE-RICH — answers should be thorough, detailed, and authoritative,
   as if written by a domain expert. Include specific names, numbers, dates.
4. NATURAL — questions should sound like a curious person or student asking.
   No mechanical "What does paragraph 3 state?" phrasing.
5. DIVERSE — each pair must teach something genuinely different. No two
   questions should be answerable with the same piece of information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD-BY-FIELD REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. knowledge_qa  (Standalone Expert Q&A — exactly 5 pairs)
   Each pair must test a DISTINCT knowledge type:

   a) FACTUAL — who / what / when / where
      Q: a specific question about a fact from the source.
      A: a direct, authoritative answer embedding the fact.
      GOOD Q: "What are the main stages of a traditional Kazakh wedding ceremony?"
      GOOD A: "A traditional Kazakh wedding (Үйлену той) consists of several
      key stages: Құда түсу (formal matchmaking between families), Қыз ұзату
      (bride's farewell ceremony), and the Үйлену той itself (the main wedding
      feast). During Құда түсу, the groom's family visits the bride's family
      to negotiate and formalize the engagement with gift exchanges."
      BAD Q: "What does the text say about weddings?"

   b) CONCEPTUAL / EXPLANATORY — what is / how does X work
      Q: asks for an explanation of a concept, tradition, or process.
      A: a thorough, multi-sentence explanation a teacher would give.
      GOOD Q: "What is the cultural significance of the dastarkhan in
      Kazakh hospitality?"
      GOOD A: "The dastarkhan is a traditional tablecloth spread on the floor
      or a low table that serves as the centerpiece of Kazakh hospitality.
      It represents generosity and respect for guests — a bare dastarkhan is
      considered deeply disrespectful. Hosts always ensure it is laden with
      bread (typically baursak), dried fruits, sweets, and tea before guests
      arrive, symbolizing abundance and welcome."

   c) COMPARATIVE — how does X relate to / differ from Y
      Q: asks for comparison between two things mentioned in the source.
      A: structured comparison with specific details from both sides.

   d) CAUSAL / ANALYTICAL — why / what caused / what resulted
      Q: asks about causes, effects, or relationships.
      A: explains the causal chain with specific evidence.

   e) APPLIED / PRACTICAL — how would / what should
      Q: asks how knowledge would be applied in practice.
      A: gives practical, actionable guidance grounded in the source.

   ALL answers: 3-8 sentences. Thorough, authoritative, standalone.
   FORBIDDEN in answers: "According to...", "The text mentions...",
   "As described in...", "Based on the passage..."

2. detailed_explanation  (In-Depth Expert Explanation)
   Pick the single richest concept, tradition, or process from the source
   and create a comprehensive teaching moment.
   • instruction: a natural request like "Explain the tradition of X in
     detail" or "Describe how Y works and why it matters."
   • response: 5-10 sentences. Multi-paragraph if needed. Structured like
     an encyclopedia entry — definition, context, details, significance.
     Must embed ALL relevant facts from the source about this topic.
   This teaches the model to produce thorough, expert-level responses.

3. analytical_reasoning  (Multi-Fact Inference)
   Create a question that requires combining 2+ facts from the source to
   reach a conclusion that is not explicitly stated.
   • instruction: a "why" or "what can we conclude" question that requires
     chaining multiple pieces of information.
   • response: MUST show the reasoning chain explicitly:
     "First, [fact A]. Second, [fact B]. Combining these, [conclusion]."
     3-6 sentences.
   This teaches the model to reason, not just recall.

4. conversational_exchange  (Natural Multi-Turn Dialogue)
   Generate a realistic 2-turn conversation where the second question is a
   natural follow-up that digs deeper.
   • opening_question: a natural first question someone would ask.
   • opening_response: a thorough, helpful answer (3-5 sentences).
   • follow_up_question: a genuine follow-up — asks for clarification,
     a related detail, or goes deeper. Must flow naturally from the first
     answer.
   • follow_up_response: a detailed response to the follow-up (3-5
     sentences), introducing NEW information from the source that wasn't
     in the first answer.
   This teaches the model to handle multi-turn dialogue naturally.

5. metadata
   • language: ISO 639-1 code of the SOURCE text (e.g. "en", "kk", "ru")
   • topics: 3-6 SPECIFIC entities, events, or concepts from the source
     (NOT generic words like "information" or "data")
   • domain: ONE of: culture, history, science, technology, politics,
     economics, art, religion, geography, cuisine, sports, education,
     health, law, other"""
