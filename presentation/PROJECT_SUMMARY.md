# Ask Syracuse Data - Project Summary

**A natural language interface for querying Syracuse Open Data**

## What It Does

Ask Syracuse Data lets anyone ask questions about Syracuse in plain English and get data-driven answers with interactive charts, maps, and honest warnings about data limitations. No coding or SQL knowledge required.

**Example:** Type *"How many code violations by neighborhood?"* and get a bar chart, bubble map, AI-generated insights, ground-truth validation, and bias warnings - all in seconds.

## Why It Matters

Syracuse publishes 16+ public datasets on data.syr.gov covering housing, crime, city services, infrastructure, and public health. But accessing this data requires technical skills most residents, journalists, and community organizations don't have. Ask Syracuse Data bridges that gap by translating everyday questions into validated data answers.

## What's Inside

- **16 datasets** - 700K+ records across housing violations, crime (2022–2025), parking tickets, 311 requests, property assessments, bike infrastructure, tree inventory, lead testing, and more
- **Hybrid architecture** - Simple queries use fast, deterministic SQL; complex queries use AI-generated SQL with safety guardrails
- **Works offline** - A keyword-based parser handles ~80% of common queries without any API key
- **Ground-truth validation** - Every result is independently verified against pandas calculations
- **Bias detection** - Automatic warnings for framing, normalization, selection, context, and uncertainty biases
- **Cross-dataset analysis** - Join violations with rental properties, crime with vacant properties, and more
- **Interactive visualizations** - Auto-generated Plotly charts (bar, line, grouped bar) and maps (neighborhood, ZIP, route lines)
- **153 automated tests** across 4 test suites

## Tech Stack

FastAPI + DuckDB + GPT-4o-mini + Tailwind CSS + Plotly.js

## Who It's For

- **Residents** checking neighborhood conditions
- **City officials** making resource allocation decisions
- **Community organizations** planning programs
- **Journalists** investigating local issues

## Try It

Visit the live demo or run locally:
```
pip install -r requirements.txt
python -m uvicorn app:app --reload
# Open http://127.0.0.1:8000
```

**GitHub:** github.com/prathyusha1231/Ask-Syracuse-Data

---
*Built by Prathyusha | Syracuse University*
