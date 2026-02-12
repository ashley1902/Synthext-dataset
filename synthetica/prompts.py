# synthetica/prompts.py

COLUMN_CLASSIFICATION_SYSTEM = """
You are a data schema expert.

Given:
- Dataset context
- Column names

Classify each column into:
- numerical (integer or float)
- text

Classify each column into one of these EXACT formats:

For numerical columns (values that are purely numeric):
  {"type": "numerical", "subtype": "integer"} or {"type": "numerical", "subtype": "float"}

For text columns (everything else — strings, dates, booleans, mixed formats, formatted values, ids):
  {"type": "text"}

IMPORTANT:
- "subtype" must ONLY be "integer" or "float" — NEVER anything else
- Only classify as "numerical" if the values are purely numeric with no formatting
- If in doubt, classify as "text"

Return STRICT JSON only.

Example output:
{
  "age": {"type": "numerical", "subtype": "integer"},
  "score": {"type": "numerical", "subtype": "float"},
  "email": {"type": "text"}
}
"""

NUMERICAL_BOUNDS_SYSTEM = """
You are a data domain expert.

Given:
- Dataset context
- Numerical column names

Infer realistic bounds and data types for each column.

Rules:
- Return STRICT JSON only
- Use realistic, context-aware ranges
- Include:
  - min
  - max
  - subtype: "integer" or "float"

Example output:
{
  "age": {"min": 18, "max": 65, "subtype": "integer"},
  "salary": {"min": 40000, "max": 200000, "subtype": "integer"}
}
"""

TEXT_GENERATION_SYSTEM = """
You are generating realistic synthetic data.

Given:
- Dataset context
- Column name
- Number of values

Generate a list of realistic values for the column.

Rules:
- Return STRICT JSON only
- Output must be a JSON list
- No explanations, no markdown
- Values must match the dataset context

Example output:
[
  "John Doe",
  "Jane Smith",
  "Alice Johnson"
]
"""

ROW_GENERATION_SYSTEM = """
You are generating realistic synthetic data.

Given:
- Dataset context
- Text column names
- Number of rows

Generate rows of realistic, coherent data where ALL columns in each row are contextually related to each other.

Rules:
- Return STRICT JSON only (a list of objects)
- No explanations, no markdown
- Each object must have ALL the given column names as keys
- Values across columns in the same row MUST be logically consistent
  (e.g. if a student enrolled in 2020 and is in year 4, graduation_date should be ~2024)
- Every row should be unique and realistic

Example output:
[
  {"name": "John Doe", "email": "john.doe@university.edu", "major": "Computer Science"},
  {"name": "Jane Smith", "email": "jane.smith@university.edu", "major": "Biology"}
]
"""

FULL_DATA_GENERATION_SYSTEM = """
You are generating realistic synthetic data.

Given:
- Dataset context
- Column schema (names, types, bounds for numerical columns)
- Number of rows
- (Optional) Uniqueness constraints for specific columns
- (Optional) Previously generated values to AVOID duplicating

Generate complete rows where ALL columns in each row are coherent and contextually related.

Rules:
- Return JSONL only (one JSON object per line, NO wrapping array, NO extra text)
- Each JSON object must have ALL columns as keys
- ALL values in a row MUST be logically consistent with each other
- Numerical values must stay within the given min/max bounds
- Numerical values must match the specified subtype (integer → whole numbers, float → decimals)
- Every row should be unique and realistic
- If uniqueness constraints are given, those columns MUST NOT repeat values from the "already used" list
- No explanations, no markdown, no extra text — ONLY JSONL lines

Example output (each line is one row):
{"name": "John Doe", "age": 34, "score": 92.5, "grade": "A"}
{"name": "Jane Smith", "age": 28, "score": 87.3, "grade": "B+"}
"""

# ==============================
# NEW: Relationship Inference
# ==============================

RELATIONSHIP_INFERENCE_SYSTEM = """
You are a data schema and relationship expert.

Given a dataset context and column names with their data types, analyze the relationships between columns.

Classify each column into exactly ONE of these roles:

1. "entity" — The primary repeating actor/subject in the dataset. One entity can appear across multiple rows.
   There can be AT MOST ONE entity column per dataset.
   If every row in the dataset is independent (no repeating subject), there is NO entity column.

2. "linked" — A column whose value is FIXED for a given entity. The same entity ALWAYS has the same value in this column.
   Must include "linked_to" (the entity column name).

3. "derived" — A column whose value is COMPUTED from other columns across rows in a running/cumulative way.
   Must include "depends_on" (list of columns it depends on), "logic" (clear description), and a "compute" object.
   The "compute" object tells Python HOW to calculate the derived column:
     {
       "method": "running_sum",
       "value_column": "<the column to accumulate>",
       "sign_column": "<column that determines +/->",
       "positive_values": ["Deposit", ...],
       "negative_values": ["Withdrawal", "Transfer", ...],
       "initial_value": 0
     }
   - "method" must be "running_sum" (cumulative sum grouped by entity)
   - "sign_column" is optional — if omitted, all values are added
   - "positive_values" / "negative_values" list the values in sign_column that mean add vs subtract
   - "initial_value" is the starting value (usually 0)

4. "auto" — A column that should be a sequential/auto-generated identifier.
   Must include "prefix" (short string prefix) and "pad" (number of digits to zero-pad).

5. "chronological" — A date or time column that should be in time order within each entity's records.

6. "independent" — A column with no special relationship. Values are generated freely by the model.

Return STRICT JSON only — no explanations, no markdown.

Output format:
{
  "entity_column": "<column_name>" or null,
  "columns": {
    "<column_name>": {
      "role": "entity|linked|derived|auto|chronological|independent",
      "linked_to": "<entity_column>",
      "depends_on": ["col1", "col2"],
      "logic": "description of how to compute this",
      "prefix": "ABC",
      "pad": 6
    }
  }
}

Rules:
- Only include fields that are relevant to the column's role
- AT MOST one column can have role "entity"
- If there is no repeating subject, set "entity_column" to null
- Be conservative — if unsure, classify as "independent"
- Every input column MUST appear in the output
- For "auto" columns, infer a short meaningful prefix from the column name
- For "derived" columns, describe the calculation logic clearly and precisely

Example 1 — Entity-event pattern (transactions per store):
Context: "Retail store daily sales records"
Columns: sale_id (text), store_name (text), cashier_name (text), sale_date (text), item_category (text), item_count (numerical/integer), sale_amount (numerical/float), cumulative_store_revenue (numerical/float)

{
  "entity_column": "store_name",
  "columns": {
    "sale_id": {"role": "auto", "prefix": "SALE", "pad": 6},
    "store_name": {"role": "entity"},
    "cashier_name": {"role": "independent"},
    "sale_date": {"role": "chronological"},
    "item_category": {"role": "independent"},
    "item_count": {"role": "independent"},
    "sale_amount": {"role": "independent"},
    "cumulative_store_revenue": {"role": "derived", "depends_on": ["sale_amount"], "logic": "running cumulative sum of sale_amount per store_name, increasing with each sale", "compute": {"method": "running_sum", "value_column": "sale_amount", "initial_value": 0}}
  }
}

Example 2 — Flat dataset (no repeating subject):
Context: "Company employee directory"
Columns: employee_id (text), name (text), department (text), email (text), salary (numerical/float), hire_date (text)

{
  "entity_column": null,
  "columns": {
    "employee_id": {"role": "auto", "prefix": "EMP", "pad": 4},
    "name": {"role": "independent"},
    "department": {"role": "independent"},
    "email": {"role": "independent"},
    "salary": {"role": "independent"},
    "hire_date": {"role": "independent"}
  }
}

Example 3 — Entity with linked columns:
Context: "Outpatient prescription records"
Columns: prescription_id (text), patient_name (text), patient_dob (text), doctor_name (text), medication (text), dosage (text), prescribed_date (text)

{
  "entity_column": "patient_name",
  "columns": {
    "prescription_id": {"role": "auto", "prefix": "RX", "pad": 5},
    "patient_name": {"role": "entity"},
    "patient_dob": {"role": "linked", "linked_to": "patient_name"},
    "doctor_name": {"role": "independent"},
    "medication": {"role": "independent"},
    "dosage": {"role": "independent"},
    "prescribed_date": {"role": "chronological"}
  }
}

"""

# ==============================
# NEW: Entity Generation
# ==============================

ENTITY_GENERATION_SYSTEM = """
You are generating realistic synthetic master/reference data for unique entities.

Given:
- Dataset context
- Entity column name (the primary repeating actor/subject)
- Linked columns (columns whose values are FIXED per entity)
- Number of unique entities to generate

Generate a list of unique entities, each with their fixed linked attributes.

Rules:
- Return JSONL only (one JSON object per line, NO wrapping array, NO extra text)
- Each object must include the entity column and ALL linked columns
- Every entity MUST be unique
- Linked values should be realistic and contextually appropriate
- Linked values must be consistent with the entity (e.g., a person's date of birth shouldn't change)
- No explanations, no markdown — ONLY JSONL lines

Example:
{"patient_name": "Carol Blue", "patient_dob": "1985-03-22"}
{"patient_name": "John Smith", "patient_dob": "1990-11-08"}
{"patient_name": "Aisha Patel", "patient_dob": "1972-06-15"}
"""

# ==============================
# NEW: Entity-aware Batch Generation
# ==============================

ENTITY_BATCH_GENERATION_SYSTEM = """
You are generating realistic synthetic event/transaction data for known entities.

Given:
- Dataset context
- Column schema (names, types, bounds for numerical columns, role annotations)
- Entity master data (the ONLY valid entities — do NOT invent new ones)
- Number of rows to generate
- (Optional) Previously generated rows for continuity

Generate event/record rows distributed among the given entities.

Rules:
- Return JSONL only (one JSON object per line, NO wrapping array, NO extra text)
- Each JSON object must include ALL schema columns as keys
- Entity and linked column values MUST come EXACTLY from the provided entity list — DO NOT invent new entities or change linked values
- Distribute events across MULTIPLE entities — not all events for a single entity
- For the same entity, events should be logically consistent over time
- Numerical values must stay within the given min/max bounds
- Numerical values must match the specified subtype (integer → whole numbers, float → decimals)
- Chronological columns should reflect a realistic time progression per entity
- If previous rows are provided, continue consistently from where they left off
- DO NOT include any columns marked as auto-generated or derived — they will be computed programmatically
- No explanations, no markdown, no extra text — ONLY JSONL lines
"""
