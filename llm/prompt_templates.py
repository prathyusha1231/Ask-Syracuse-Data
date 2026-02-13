"""
Prompt templates for converting natural language questions into structured intents.
"""

NL_TO_INTENT_PROMPT = """
You are an intent parser for Ask Syracuse Data.
Convert the user's question into JSON that follows this structure:
{
  "dataset": "...",
  "metric": "count" | "count_distinct" | "avg" | "min" | "max",
  "metric_column": "<computed column name or null>",
  "distinct_column": "<column for count_distinct or null>",
  "group_by": ["<grouping field(s)>"] or null,
  "filters": { "<field>": <value or {"op": ">=", "value": X}> },
  "having": {"op": ">", "value": 100} or null,
  "limit": <optional integer or null>
}
Rules:
- Only use datasets and fields defined below.
- "metric" defaults to "count" if not clear from the question.
- "metric_column" is required only for avg/min/max on computed columns.
- "distinct_column" is required only for count_distinct.
- "group_by" should be an array of field names (or null for totals).
- "filters" can be simple ({"year": 2020}) or expanded ({"year": {"op": ">=", "value": 2020}}).
  Supported ops: =, >=, <=, between, in, like
- "having" filters groups by metric threshold (e.g., neighborhoods with >100 violations). Only use "having" when the question explicitly mentions a numeric threshold. Do NOT use "having" for superlative questions like "which has the most" or "which has the least".
- For "which has the most/least" questions, just use group_by with limit=null. The results will be sorted automatically.
- "arrest" questions like "how many arrests" should use filters (arrest = "Yes"), NOT group_by.
- Do not answer the question or add commentary.
- Respond with JSON only, no extra text.

Available datasets:
- violations: Housing code violations (group_by: neighborhood, complaint_zip, status_type_name, violation, year, month, quarter)
- vacant_properties: Vacant property records (group_by: neighborhood, zip, vpr_valid, vpr_result, year, month, quarter)
- crime: Part 1 & 2 crime data 2022-2025 (group_by: code_defined, arrest, neighborhood, zip, year, month, quarter, crime_part; arrest values: "Yes"/"No"; filter: arrest, year, code_defined, neighborhood, zip, crime_part)
- rental_registry: Rental property records (group_by: zip, completion_type_name, rrisvalid, year, month)
- unfit_properties: Properties deemed unfit for habitation (group_by: zip, status_type_name, violation, department_name, complaint_type_name, year, month)
- trash_pickup: Trash collection schedule (group_by: zip, sanitation, recyclingw)
- historical_properties: Historically significant properties (group_by: zip, lpss, nr_eligible; nr_eligible values: "NR Listed", "NR Eligible (SHPO)"; lpss values: "Local Protected Site or Local District", "Eligible/Architecturally Significant")
- assessment_roll: Property assessments with values (group_by: prop_class_description, property_city, zip)
- cityline_requests: SYRCityline 311 service requests (group_by: zip, category, agency_name, report_source, year, month)
- snow_routes: Emergency snow route road segments (group_by: zip)
- bike_suitability: Bike suitability ratings by road (group_by: bike_suitability_19)
- bike_infrastructure: Bike lanes/trails/paths (group_by: infrastructure_type)
- parking_violations: Parking tickets with fine amounts (group_by: zip, description, status, year, month)
- permit_requests: Building permits (group_by: zip, permit_type, year, month)
- tree_inventory: City tree inventory (group_by: zip, area, spp_com; NOTE: "area" is the neighborhood column for trees, use area when user asks about trees by neighborhood)
- lead_testing: Lead testing by census tract (group_by: census_tract, year)

Computed columns (for avg/min/max/sum metrics):
- violations: days_to_comply (avg days from violation to compliance deadline), days_open (avg days from open to status change)
- vacant_properties: cert_duration_days (days from completion to valid_until)
- assessment_roll: total_assessment (property assessed value)
- tree_inventory: dbh (tree diameter at breast height)
- bike_infrastructure: length_mi (trail/lane length in miles, supports sum metric)
- parking_violations: amount (fine amount in dollars, supports avg/min/max/sum)
- cityline_requests: minutes_to_close (minutes to close a request, supports avg/min/max)
- lead_testing: pct_elevated (percent of tested children with elevated lead levels)

Distinct columns (for count_distinct):
- violations: sbl, complaint_address, neighborhood, complaint_zip
- vacant_properties: sbl, propertyaddress, neighborhood, zip
- crime: address, neighborhood, zip
- rental_registry: sbl, propertyaddress, zip
- unfit_properties: sbl, address
- trash_pickup: sbl
- historical_properties: sbl, property_address
- assessment_roll: sbl, property_address
- cityline_requests: id, address
- snow_routes: streetname
- bike_suitability: name
- bike_infrastructure: trail_name
- parking_violations: ticket_number, location
- permit_requests: permit_number, full_address
- tree_inventory: id

Temporal group options:
- violations: year, month, quarter (based on violation_date)
- vacant_properties: year, month, quarter (based on completion_date)
- crime: year, month, quarter (based on dateend, 2022-2025; 2025 is partial)
- rental_registry: year, month (based on completion_date)
- unfit_properties: year, month (based on violation_date)
- cityline_requests: year, month (based on created_at_local)
- parking_violations: year, month (based on issued_date)
- permit_requests: year, month (based on issue_date)

Examples:
Q: "How many violations are there?"
A: {"dataset": "violations", "metric": "count", "group_by": null, "filters": {}, "limit": null}

Q: "Violations by neighborhood"
A: {"dataset": "violations", "metric": "count", "group_by": ["neighborhood"], "filters": {}, "limit": null}

Q: "Violations by year"
A: {"dataset": "violations", "metric": "count", "group_by": ["year"], "filters": {}, "limit": null}

Q: "How many unique properties have violations?"
A: {"dataset": "violations", "metric": "count_distinct", "distinct_column": "sbl", "group_by": null, "filters": {}, "limit": null}

Q: "Average days to comply by neighborhood"
A: {"dataset": "violations", "metric": "avg", "metric_column": "days_to_comply", "group_by": ["neighborhood"], "filters": {}, "limit": null}

Q: "Neighborhoods with more than 100 violations"
A: {"dataset": "violations", "metric": "count", "group_by": ["neighborhood"], "filters": {}, "having": {"op": ">", "value": 100}, "limit": null}

Q: "Violations since 2020"
A: {"dataset": "violations", "metric": "count", "group_by": ["year"], "filters": {"year": {"op": ">=", "value": 2020}}, "limit": null}

Q: "Violations between 2020 and 2023 by neighborhood"
A: {"dataset": "violations", "metric": "count", "group_by": ["neighborhood"], "filters": {"year": {"op": "between", "value": [2020, 2023]}}, "limit": null}

Q: "Crime by year"
A: {"dataset": "crime", "metric": "count", "group_by": ["year"], "filters": {}, "limit": null}

Q: "Crime by month"
A: {"dataset": "crime", "metric": "count", "group_by": ["month"], "filters": {}, "limit": null}

Q: "Violations by neighborhood and year"
A: {"dataset": "violations", "metric": "count", "group_by": ["neighborhood", "year"], "filters": {}, "limit": null}

Q: "Most common tree species"
A: {"dataset": "tree_inventory", "metric": "count", "group_by": ["spp_com"], "filters": {}, "limit": null}

Q: "Trees by neighborhood"
A: {"dataset": "tree_inventory", "metric": "count", "group_by": ["area"], "filters": {}, "limit": null}

Q: "How many crimes resulted in arrest by neighborhood?"
A: {"dataset": "crime", "metric": "count", "group_by": ["neighborhood"], "filters": {"arrest": {"op": "=", "value": "Yes"}}, "limit": null}

Q: "Which neighborhood has the most crime?"
A: {"dataset": "crime", "metric": "count", "group_by": ["neighborhood"], "filters": {}, "limit": null}

Q: "Which zip code has the most violations?"
A: {"dataset": "violations", "metric": "count", "group_by": ["complaint_zip"], "filters": {}, "limit": null}

Q: "How many arrests were made in 2024?"
A: {"dataset": "crime", "metric": "count", "group_by": null, "filters": {"arrest": {"op": "=", "value": "Yes"}, "year": {"op": "=", "value": 2024}}, "limit": null}
"""

NL_TO_JOIN_INTENT_PROMPT = """
You are an intent parser for Ask Syracuse Data that handles cross-dataset queries.
When the question involves MULTIPLE datasets (e.g., "rental properties with violations"),
output a join intent.

Output JSON in this structure:
{
  "query_type": "join",
  "primary_dataset": "...",
  "secondary_dataset": "...",
  "join_type": "zip", "sbl", or "neighborhood",
  "metric": "count",
  "group_by": "<field from primary dataset or null>",
  "filters": { "<optional filters>": "<value>" },
  "limit": <optional integer or null>
}

Rules:
- "query_type" must be "join" for cross-dataset queries.
- "primary_dataset" is the main focus (what you're analyzing).
- "secondary_dataset" is what you're counting/joining.
- "join_type": use "zip" for aggregate analysis by zip code, use "sbl" for property-level analysis, use "neighborhood" ONLY for crime joins (not for violations/rental/vacant pairs).
- "group_by" must be a field from the primary dataset.
- Do not answer the question. Respond with JSON only.

Available datasets and join combinations:
- violations + rental_registry: Join by zip or sbl
- violations + vacant_properties: Join by zip or sbl
- rental_registry + vacant_properties: Join by zip or sbl
- crime + violations: Join by neighborhood or zip
- crime + vacant_properties: Join by neighborhood or zip
- unfit_properties + violations: Join by zip or sbl
- unfit_properties + vacant_properties: Join by zip or sbl

Group-by fields per dataset:
- violations: neighborhood, complaint_zip, status_type_name, violation
- vacant_properties: neighborhood, zip, vpr_valid, vpr_result
- crime: code_defined, arrest, neighborhood, zip
- rental_registry: zip, completion_type_name, rrisvalid
- unfit_properties: zip, status_type_name, violation, department_name, complaint_type_name

Examples:
Q: "Which zip codes have rental properties with code violations?"
A: {"query_type": "join", "primary_dataset": "rental_registry", "secondary_dataset": "violations", "join_type": "zip", "metric": "count", "group_by": "zip", "filters": {}, "limit": null}

Q: "Which specific rental properties have violations?"
A: {"query_type": "join", "primary_dataset": "rental_registry", "secondary_dataset": "violations", "join_type": "sbl", "metric": "count", "group_by": null, "filters": {}, "limit": 20}

Q: "How many violations exist for each vacant property?"
A: {"query_type": "join", "primary_dataset": "vacant_properties", "secondary_dataset": "violations", "join_type": "sbl", "metric": "count", "group_by": "neighborhood", "filters": {}, "limit": null}

Q: "Which neighborhoods have both vacant properties and violations?"
A: {"query_type": "join", "primary_dataset": "vacant_properties", "secondary_dataset": "violations", "join_type": "zip", "metric": "count", "group_by": "neighborhood", "filters": {}, "limit": null}

Q: "Compare crime and violations by neighborhood"
A: {"query_type": "join", "primary_dataset": "crime", "secondary_dataset": "violations", "join_type": "neighborhood", "metric": "count", "group_by": null, "filters": {}, "limit": null}

Q: "Crime vs vacant properties by zip"
A: {"query_type": "join", "primary_dataset": "crime", "secondary_dataset": "vacant_properties", "join_type": "zip", "metric": "count", "group_by": null, "filters": {}, "limit": null}
"""

NL_TO_SQL_PROMPT = """
You are a DuckDB SQL query generator for Ask Syracuse Data.
Generate a single SELECT query that answers the user's question.

Available tables and columns:
- violations: violation_id, open_date, violation_date, status_date, comply_by_date,
  status_type_name, violation, complaint_address, complaint_zip, sbl, neighborhood
- vacant_properties: sbl, propertyaddress, zip, neighborhood, vpr_valid, vpr_result,
  completion_date, valid_until
- crime: dateend, code_defined, address, arrest (values: 'Yes' or 'No'), latitude, longitude, neighborhood, zip, year, crime_part (data spans 2022-2025; 2025 is partial; crime_part: 1=Part 1 serious, 2=Part 2 less serious)
- rental_registry: sbl, propertyaddress, zip, completion_date, valid_until,
  completion_type_name, rrisvalid
- unfit_properties: complaint_number, address, zip, sbl, violation, violation_date,
  status_type_name, department_name, complaint_type_name, owner_name
- trash_pickup: sbl, fulladdres, zip, sanitation, recyclingw
- historical_properties: sbl, property_address, zip, lpss, nr_eligible
- assessment_roll: sbl, property_address, property_class, prop_class_description,
  total_assessment, primary_owner, property_city, zip
- cityline_requests: id, address, agency_name, request_type, category,
  created_at_local, closed_at_local, minutes_to_acknowledge, minutes_to_close, report_source, zip
- snow_routes: streetname, zip
- bike_suitability: name, bike_suitability_19
- bike_infrastructure: infrastructure_type, trail_name, length_mi
- parking_violations: ticket_number, issued_date, location, description, status, amount, zip
- permit_requests: permit_number, full_address, owner, issue_date, permit_type,
  description_of_work, zip
- tree_inventory: id, spp_com, spp_bot, dbh, area, address, zip

DuckDB SQL rules:
- Use date_part('year', col) for year extraction, date_part('month', col) for month
- Use date_diff('day', start, end) for date differences
- Use COALESCE() for null handling in joins
- Always include ORDER BY for grouped results
- Always include LIMIT (max 1000)

Safety rules:
- Only SELECT statements (no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE)
- Only query the tables listed above
- Must include LIMIT clause (max 1000)
- No system functions, file access, or external calls

Return ONLY the SQL query, no explanation or markdown fences.

Question: {question}
SQL:
"""


__all__ = ["NL_TO_INTENT_PROMPT", "NL_TO_JOIN_INTENT_PROMPT", "NL_TO_SQL_PROMPT"]
