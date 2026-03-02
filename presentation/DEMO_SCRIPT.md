# Demo Script - Ask Syracuse Data (10 minutes)

## Setup Before Demo
1. Start server: `python -m uvicorn app:app --reload`
2. Open browser to http://127.0.0.1:8000
3. Have slides open in separate window

---

## The Problem (Slides 1-3) - ~2 minutes

**[Slide 1 - Title]**
"Today I'm presenting Ask Syracuse Data - a natural language interface that lets anyone query Syracuse's public data using plain English."

**[Slide 2 - The Problem]**
"Syracuse publishes 16 datasets on data.syr.gov - housing violations, crime, parking tickets, city services, and more. But this data is locked behind CSV downloads and technical jargon. The people who need it most - residents, journalists, community organizations - can't access it without SQL or Python skills."

**[Slide 3 - Why Syracuse Should Care]**
"We cover 16 datasets with over 700,000 records. A resident can ask about violations in their ZIP code. A city official can cross-reference rental properties with code violations. A journalist can look at crime trends over time."

---

## The Approach (Slide 4-5) - ~2 minutes

**[Slide 4 - Architecture]**
"The system uses a hybrid architecture. When you ask a question, it goes through two possible paths:
- Path 1 handles simple and medium queries deterministically - no hallucination risk, works offline, fully testable.
- Path 2 handles complex queries like rankings and percentiles using LLM-generated SQL, but with strict safety guardrails - read-only enforcement, table allowlist, and a 1000-row limit.

The key insight: the LLM never sees your data. It only translates language into structure."

**[Slide 5 - Validation]**
"What makes this different from just asking ChatGPT? Every result is independently validated against a pandas ground-truth calculation. We also run 5 types of bias detection - framing, normalization, selection, context, and uncertainty - so users know what the data can and can't tell them."

---

## The Findings / Live Demo (Slide 6) - ~4 minutes

**[Switch to browser - http://127.0.0.1:8000]**

### Demo Query 1: Simple count with grouping
Type: **"How many code violations by neighborhood?"**
- Point out: bar chart auto-generated, bubble map shows geographic distribution
- Click **Insights** tab: AI-generated explanation
- Click **Validation** tab: ground-truth comparison (pandas count matches SQL count)
- Click **Sources** tab: bias warnings + data citation

### Demo Query 2: Computed metric
Type: **"Average parking fine by type"**
- Show: ranked bar chart of fine amounts by violation type
- Note: this uses a computed column (amount parsed from mixed formats)

### Demo Query 3: Cross-dataset join
Type: **"Rental properties with violations by ZIP"**
- Show: cross-dataset join combining rental registry + code violations
- Note: uses SBL (property identifier) to link records across datasets

### Demo Query 4: Temporal trend
Type: **"Crime by year"**
- Show: line chart with 2022-2024 trend
- Point out bias warning about 2025 being partial data (33 records)

### Demo Query 5 (if time): Feedback
- Click thumbs up/down on any result
- Mention: feedback stored in SQLite for tracking query quality

**[Switch back to slides]**

---

## Limitations & Future Work (Slide 7-8) - ~2 minutes

**[Slide 7 - Testing]**
"We have 153 automated tests across 4 test suites. The heuristic parser handles about 80% of common queries without any API key, making the app functional offline."

**[Slide 8 - Limitations]**
"To be transparent about what we can't do yet:
- This is static data, not live feeds
- We report correlations, not causes - a neighborhood with more violations may just have more active enforcement
- Raw counts aren't normalized by population
- And we explicitly decline subjective questions like 'what's the safest neighborhood?'

For future work, I'd love to add live data integration, per-capita normalization, multi-turn conversation, and Spanish language support."

**[Slide 9 - Thank You]**
"Thank you. The code is on GitHub and I'm happy to do a deeper dive into any part of the system. Questions?"

---

## Backup Queries (if questions arise)
- "How many unique properties have violations?" - count_distinct
- "Neighborhoods with more than 100 violations" - HAVING threshold
- "Lead testing by census tract" - public health data
- "Bike infrastructure by type" - infrastructure with route line map
- "Trees by neighborhood" - tree inventory (auto-aliases to "area" column)
