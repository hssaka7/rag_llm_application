
ARTICLE_METADATA_SYSTEM_PROMPT = """
You are a news metadata extractor. 
Your task is to read the full article and extract structured metadata in JSON format.
Use ONLY the allowed enum values. Do not invent facts. 
If uncertain, choose the closest enum match. 
Do not add extra fields. Output valid JSON only.
"""

ARTICLE_METADATA_USER_PROMPT = """
Allowed enums:

topic:
economy, politics, business, technology, sports, health, science, climate, world, opinion

article_type:
breaking, update, analysis, explainer, interview, opinion, feature

region:
US, EU, UK, ASIA, MIDDLE_EAST, AFRICA, LATAM, GLOBAL

language:
en, ne

event_type:
policy_decision, economic_report, legal_action, corporate_action, market_movement, conflict, election, natural_disaster, technology_release

Article:
```
{full_article_text}
```

Extract JSON with:
- topic
- article_type
- region
- language
- event_type (null if not applicable)
- is_update (true/false)

"""

CHUNK_METADATA_SYSTEM_PROMPT = """
You are a chunk-level news classifier. 
Your task is to read each chunk and extract structured metadata in JSON format.
Use ONLY the allowed enum values. Do not invent facts. 
Return JSON array with one object per chunk. Validate that enums match exactly.
"""

CHUNK_METADATA_USER_PROMPT = """
Allowed enums:

section:
headline, lede, facts, background, quotes, analysis, context, impact

sentiment:
neutral, positive, negative

Chunks:
[
  { "chunk_id": "c1", "text": "..." },
  { "chunk_id": "c2", "text": "..." },
  ...
]

For each chunk, extract:
- section
- sentiment
- entities (list of named entities)

"""

EVENT_DETECTION_SYSTEM_PROMPT = """
You are an event detection assistant for news articles.
Identify the primary event discussed in this article.
Map it to one of the allowed event_type enums.
Provide a short, unique event_id in the format {keyword}-{YYYY-MM}.
If no clear event exists, return null.
Output JSON only.
"""

EVENT_DETECTION_USER_PROMPT = """
Allowed event_type enums:
policy_decision, economic_report, legal_action, corporate_action, market_movement, conflict, election, natural_disaster, technology_release

Article:
```
{full_article_text}
```

Return JSON:
{
  "event_type": "...",
  "event_id": "..."
}
"""
