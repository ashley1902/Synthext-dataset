"""
Incremental pipeline test — n+1 testing
Run with: python3 tests/test_pipeline.py

Each step builds on the previous one.
Set STOP_AFTER to control how far to go:
  1 = Column Classification only
  2 = + Numerical Bounds
  3 = + Relationship Inference
  4 = + Entity Generation
  5 = + One batch of entity events + post-processing (derived + auto)
  6 = + Full generation (all batches + post-processing)
"""

import sys
import json

STOP_AFTER = int(sys.argv[1]) if len(sys.argv) > 1 else 6

# ── Setup ──────────────────────────────────────────────

from synthetica.generator import DatasetGenerator

gen = DatasetGenerator(
    context="E-commerce order records for a mid-sized online retail platform selling electronics and accessories",
    columns=[
        "order_id",
        "customer_email",
        "customer_name",
        "order_date",
        "product_category",
        "product_name",
        "quantity",
        "unit_price",
        "order_total",
        "cumulative_customer_spend",
        "payment_method",
        "shipping_city"
    ],
    size=1000,
    batch_size=50
)

print("=" * 60)
print(f"  PIPELINE TEST — running steps 1 through {STOP_AFTER}")
print("=" * 60)


# ── Step 1: Column Classification ──────────────────────

print("\n\n" + "▓" * 60)
print("  STEP 1: Column Classification")
print("▓" * 60)

specs = gen._classify_columns()

print("\n── Result ──")
for s in specs:
    print(f"  {s.name}: type={s.type}, subtype={s.subtype}")

if STOP_AFTER == 1:
    print("\n✅ Stopped after Step 1")
    sys.exit(0)


# ── Step 2: Numerical Bounds ──────────────────────────

print("\n\n" + "▓" * 60)
print("  STEP 2: Numerical Bounds Inference")
print("▓" * 60)

specs = gen._infer_numerical_bounds(specs)

print("\n── Result ──")
for s in specs:
    if s.type == "numerical":
        print(f"  {s.name}: subtype={s.subtype}, min={s.min}, max={s.max}")
    else:
        print(f"  {s.name}: text (no bounds)")

if STOP_AFTER == 2:
    print("\n✅ Stopped after Step 2")
    sys.exit(0)


# ── Step 3: Relationship Inference ─────────────────────

print("\n\n" + "▓" * 60)
print("  STEP 3: Relationship Inference")
print("▓" * 60)

relationships = gen._infer_relationships(specs)

print("\n── Result ──")
print(f"  Entity column: {relationships.entity_column}")
for col_name, role in relationships.columns.items():
    extra = ""
    if role.role == "linked":
        extra = f" → linked_to='{role.linked_to}'"
    elif role.role == "derived":
        extra = f" → depends_on={role.depends_on}, logic='{role.logic}'"
        if role.compute:
            extra += f"\n    compute: {json.dumps(role.compute)}"
    elif role.role == "auto":
        extra = f" → prefix='{role.prefix}', pad={role.pad}"
    print(f"  {col_name}: {role.role}{extra}")

print(f"\n  Mode: {'ENTITY' if relationships.entity_column else 'SIMPLE'}")

if STOP_AFTER == 3:
    print("\n✅ Stopped after Step 3")
    sys.exit(0)


# ── Step 4: Entity Generation ──────────────────────────

print("\n\n" + "▓" * 60)
print("  STEP 4: Entity Generation")
print("▓" * 60)

if relationships.entity_column:
    entities = gen._generate_entities(relationships)

    print("\n── Result ──")
    print(f"  Generated {len(entities)} entities:")
    for e in entities:
        print(f"    {json.dumps(e)}")
else:
    entities = None
    print("\n  ⏭ Skipped — no entity column detected (simple mode)")

if STOP_AFTER == 4:
    print("\n✅ Stopped after Step 4")
    sys.exit(0)


# ── Step 5: Single Batch + Post-processing ─────────────

print("\n\n" + "▓" * 60)
print("  STEP 5: Single Batch (10 rows) + Post-processing")
print("▓" * 60)

if relationships.entity_column and entities:
    schema_str = gen._build_entity_schema_str(specs, relationships)

    print(f"\n── Schema (sent to LLM) ──")
    print(schema_str)

    # Show derivation rules (for display only — not sent to LLM)
    derivation_rules = gen._build_derivation_rules(relationships)
    if derivation_rules:
        print(f"\n── Derivation Rules (computed in Python) ──")
        print(derivation_rules)

    print(f"\n── Generating 10 rows via LLM... ──")
    rows = gen._generate_entity_batch(
        specs, schema_str, entities,
        count=10, all_rows=[], relationships=relationships, batch_num=1
    )

    print(f"\n── Raw LLM rows: {len(rows)} (before post-processing) ──")
    for i, row in enumerate(rows):
        print(f"  Row {i+1}: {json.dumps(row)}")

    # Now post-process: sort → compute derived → add auto
    print(f"\n── Post-processing... ──")
    rows = gen._post_process(rows, relationships)

    print(f"\n── Final rows: {len(rows)} (after post-processing) ──")
    for i, row in enumerate(rows):
        print(f"  Row {i+1}: {json.dumps(row)}")

else:
    schema_str = gen._build_schema_str(specs)
    rows = gen._generate_batch(
        specs, schema_str, count=10, all_rows=[], used_values={}, batch_num=1
    )

    print(f"\n── Result: {len(rows)} rows ──")
    for i, row in enumerate(rows):
        print(f"  Row {i+1}: {json.dumps(row)}")

if STOP_AFTER == 5:
    print("\n✅ Stopped after Step 5")
    sys.exit(0)


# ── Step 6: Full Generation ───────────────────────────

print("\n\n" + "▓" * 60)
print(f"  STEP 6: Full Generation ({gen.size} rows, batched)")
print("▓" * 60)

df = gen.generate()

print("\n── Result ──")
print(f"  Shape: {df.shape}")
print(f"\n  Describe:")
print(df.describe(include="all"))
print(f"\n  First 20 rows:")
print(df.head(20).to_string())

# Show derived column verification for one entity
if relationships.entity_column:
    entity_col = relationships.entity_column
    first_entity = df[entity_col].iloc[0]
    entity_rows = df[df[entity_col] == first_entity].head(10)
    
    # Find derived columns to display
    derived_cols = [c for c, r in relationships.columns.items() if r.role == "derived"]
    if derived_cols:
        print(f"\n── Derived column verification for '{first_entity}' ──")
        for _, row in entity_rows.iterrows():
            parts = []
            for col in df.columns:
                if col in (entity_col,) or col in derived_cols:
                    continue
                if col in ("order_total", "unit_price", "quantity", "transaction_amount"):
                    parts.append(f"{col}={row.get(col, '?')}")
            derived_parts = [f"{dc}={row.get(dc, '?')}" for dc in derived_cols]
            print(f"  {' | '.join(parts)}  →  {' | '.join(derived_parts)}")

df.to_csv("test_pipeline_output.csv", index=False)
print("\n  Saved to test_pipeline_output.csv")

print("\n✅ Full pipeline complete!")
