import json
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from synthetica.llm import OpenAIClient
from synthetica.models import ColumnSpec, ColumnRole, RelationshipSpec
from synthetica.prompts import (
    COLUMN_CLASSIFICATION_SYSTEM,
    NUMERICAL_BOUNDS_SYSTEM,
    TEXT_GENERATION_SYSTEM,
    ROW_GENERATION_SYSTEM,
    FULL_DATA_GENERATION_SYSTEM,
    RELATIONSHIP_INFERENCE_SYSTEM,
    ENTITY_GENERATION_SYSTEM,
    ENTITY_BATCH_GENERATION_SYSTEM
)


class DatasetGenerator:
    def __init__(
        self,
        context: str,
        columns: List[str],
        size: int,
        openai_api_key: str | None = None,
        unique_columns: Optional[List[str]] = None,
        batch_size: int = 50,
        context_window: int = 20
    ):
        self.context = context
        self.columns = columns
        self.size = size
        self.llm = OpenAIClient(openai_api_key)
        self.unique_columns = unique_columns or []
        self.batch_size = batch_size
        self.context_window = context_window

    # =========================================================
    # Phase 1: Column Classification
    # =========================================================
    def _classify_columns(self) -> List[ColumnSpec]:
        user_prompt = f"""
Dataset context:
{self.context}

Columns:
{self.columns}
"""

        raw = self.llm.complete(
            COLUMN_CLASSIFICATION_SYSTEM,
            user_prompt
        )

        print("\n=== RAW LLM OUTPUT (CLASSIFICATION) ===")
        print(raw)

        parsed = json.loads(raw)

        # Sanitize: fix invalid type/subtype combinations
        for k, v in parsed.items():
            if v.get("type") == "numerical" and v.get("subtype") not in ("integer", "float"):
                print(f"  [AUTO-FIX] '{k}' had type=numerical but subtype='{v.get('subtype')}' → reclassified as text")
                v["type"] = "text"
                v.pop("subtype", None)

        return [
            ColumnSpec(name=k, **v)
            for k, v in parsed.items()
        ]

    def debug_classification(self):
        specs = self._classify_columns()
        print("\n=== COLUMN CLASSIFICATION DEBUG ===")
        for spec in specs:
            print(spec)

    # =========================================================
    # Phase 2: Numerical Bounds Inference
    # =========================================================
    def _infer_numerical_bounds(
        self,
        specs: List[ColumnSpec]
    ) -> List[ColumnSpec]:

        numerical_specs = [s for s in specs if s.type == "numerical"]
        if not numerical_specs:
            return specs

        user_prompt = f"""
Dataset context:
{self.context}

Numerical columns:
{[s.name for s in numerical_specs]}
"""

        raw = self.llm.complete(
            NUMERICAL_BOUNDS_SYSTEM,
            user_prompt
        )

        print("\n=== RAW LLM OUTPUT (BOUNDS) ===")
        print(raw)

        bounds = json.loads(raw)

        for spec in specs:
            if spec.name in bounds:
                spec.min = bounds[spec.name]["min"]
                spec.max = bounds[spec.name]["max"]
                spec.subtype = bounds[spec.name]["subtype"]

        return specs

    def debug_bounds(self):
        specs = self._classify_columns()
        specs = self._infer_numerical_bounds(specs)
        print("\n=== NUMERICAL BOUNDS DEBUG ===")
        for spec in specs:
            print(spec)

    # =========================================================
    # Phase 3: Relationship Inference (NEW)
    # =========================================================
    def _infer_relationships(self, specs: List[ColumnSpec]) -> RelationshipSpec:
        """Use LLM to analyze column relationships and roles."""

        spec_summary = []
        for s in specs:
            if s.type == "numerical":
                spec_summary.append(f"- {s.name}: {s.type} ({s.subtype})")
            else:
                spec_summary.append(f"- {s.name}: {s.type}")

        user_prompt = f"""
Dataset context:
{self.context}

Columns and their types:
{chr(10).join(spec_summary)}
"""

        try:
            raw = self.llm.complete(
                RELATIONSHIP_INFERENCE_SYSTEM,
                user_prompt
            )

            print("\n=== RAW LLM OUTPUT (RELATIONSHIPS) ===")
            print(raw)

            parsed = json.loads(raw)

            # --- Sanitize ---
            entity_col = parsed.get("entity_column")
            columns_raw = parsed.get("columns", {})

            # Validate entity_column exists in our columns
            if entity_col and entity_col not in self.columns:
                print(f"  [AUTO-FIX] entity_column '{entity_col}' not in columns → set to null")
                entity_col = None

            # Parse column roles with validation
            valid_roles = {"entity", "linked", "derived", "auto", "chronological", "independent"}
            columns: Dict[str, ColumnRole] = {}

            for col_name in self.columns:
                if col_name in columns_raw:
                    role_data = columns_raw[col_name]
                    role = role_data.get("role", "independent")

                    if role not in valid_roles:
                        print(f"  [AUTO-FIX] '{col_name}' had invalid role '{role}' → set to 'independent'")
                        role = "independent"

                    # If entity was nullified but a column references it as linked, fix it
                    if role == "linked" and entity_col is None:
                        print(f"  [AUTO-FIX] '{col_name}' was 'linked' but no entity exists → set to 'independent'")
                        role = "independent"

                    columns[col_name] = ColumnRole(
                        role=role,
                        linked_to=role_data.get("linked_to") if role == "linked" else None,
                        depends_on=role_data.get("depends_on") if role == "derived" else None,
                        logic=role_data.get("logic") if role == "derived" else None,
                        compute=role_data.get("compute") if role == "derived" else None,
                        prefix=role_data.get("prefix") if role == "auto" else None,
                        pad=role_data.get("pad") if role == "auto" else None
                    )
                else:
                    # Column not in LLM response → default to independent
                    print(f"  [AUTO-FIX] '{col_name}' missing from LLM response → set to 'independent'")
                    columns[col_name] = ColumnRole(role="independent")

            result = RelationshipSpec(entity_column=entity_col, columns=columns)

            # Print summary
            print("\n=== INFERRED RELATIONSHIPS ===")
            print(f"  Entity column: {result.entity_column}")
            for col_name, role in result.columns.items():
                extra = ""
                if role.role == "linked":
                    extra = f" → linked to '{role.linked_to}'"
                elif role.role == "derived":
                    extra = f" → depends on {role.depends_on} | logic: {role.logic}"
                elif role.role == "auto":
                    extra = f" → prefix='{role.prefix}', pad={role.pad}"
                print(f"  • {col_name}: {role.role}{extra}")

            return result

        except Exception as e:
            print(f"\n  ⚠ Relationship inference failed: {e}")
            print(f"    Falling back to simple generation mode.")
            return RelationshipSpec(
                entity_column=None,
                columns={col: ColumnRole(role="independent") for col in self.columns}
            )

    # =========================================================
    # Shared Utilities
    # =========================================================
    def _parse_jsonl(self, raw: str) -> List[dict]:
        """Parse JSONL string into list of dicts."""
        rows = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [SKIP] Could not parse line: {line[:80]}...")
        return rows

    def _build_schema_str(self, specs: List[ColumnSpec]) -> str:
        """Build column schema string for simple mode."""
        schema_lines = []
        for s in specs:
            if s.type == "numerical":
                schema_lines.append(
                    f"- {s.name}: {s.subtype}, min={s.min}, max={s.max}"
                )
            else:
                schema_lines.append(f"- {s.name}: text")
        return "\n".join(schema_lines)

    # =========================================================
    # Entity Generation (NEW)
    # =========================================================
    def _determine_entity_count(self) -> int:
        """Determine how many unique entities to generate."""
        count = max(self.size // 7, 5)
        count = min(count, 200)
        return count

    def _generate_entities(self, relationships: RelationshipSpec) -> List[dict]:
        """Generate unique entity master data via LLM."""
        entity_col = relationships.entity_column
        linked_cols = [
            col_name for col_name, role in relationships.columns.items()
            if role.role == "linked"
        ]

        num_entities = self._determine_entity_count()

        user_prompt = f"""
Dataset context:
{self.context}

Entity column: {entity_col}
Linked columns (values that are FIXED per entity): {linked_cols if linked_cols else "none"}

Number of unique entities to generate: {num_entities}
"""

        raw = self.llm.complete(ENTITY_GENERATION_SYSTEM, user_prompt)

        print(f"\n=== RAW LLM OUTPUT (ENTITIES: {num_entities} requested) ===")
        print(raw)

        entities = self._parse_jsonl(raw)

        if not entities:
            raise ValueError("LLM returned no entities")

        print(f"\n=== GENERATED {len(entities)} ENTITIES ===")
        for e in entities[:5]:
            print(f"  {e}")
        if len(entities) > 5:
            print(f"  ... and {len(entities) - 5} more")

        return entities

    # =========================================================
    # Entity-mode Batch Generation (NEW)
    # =========================================================
    def _build_derivation_rules(self, relationships: RelationshipSpec) -> str:
        """Build human-readable derivation rules string."""
        rules = []
        for col_name, role in relationships.columns.items():
            if role.role == "derived":
                deps = ", ".join(role.depends_on) if role.depends_on else "other columns"
                rules.append(f"- {col_name}: {role.logic} (depends on: {deps})")
        return "\n".join(rules) if rules else ""

    def _build_entity_schema_str(
        self,
        specs: List[ColumnSpec],
        relationships: RelationshipSpec
    ) -> str:
        """Build column schema string for entity mode, excluding auto and derived columns."""
        schema_lines = []
        for s in specs:
            role = relationships.columns.get(s.name)

            # Skip auto columns — they'll be added in post-processing
            if role and role.role == "auto":
                continue

            # Skip derived columns — they'll be computed in post-processing
            if role and role.role == "derived":
                continue

            # Build base type info
            if s.type == "numerical":
                base = f"- {s.name}: {s.subtype}, min={s.min}, max={s.max}"
            else:
                base = f"- {s.name}: text"

            # Add role annotation for the LLM's awareness
            if role:
                if role.role == "entity":
                    base += "  [ENTITY — main repeating subject]"
                elif role.role == "linked":
                    base += f"  [LINKED to {role.linked_to} — same value for same entity]"
                elif role.role == "chronological":
                    base += "  [CHRONOLOGICAL — in time order per entity]"

            schema_lines.append(base)
        return "\n".join(schema_lines)

    def _build_entity_batch_prompt(
        self,
        schema_str: str,
        entities: List[dict],
        count: int,
        all_rows: List[dict],
        relationships: RelationshipSpec
    ) -> str:
        """Build the user prompt for entity-aware batch generation."""

        prompt = f"""
Dataset context:
{self.context}

Column schema:
{schema_str}

Entity master data (use ONLY these entities — DO NOT create new ones):
"""
        for entity in entities:
            prompt += json.dumps(entity) + "\n"

        chrono_cols = [
            col_name for col_name, role in relationships.columns.items()
            if role.role == "chronological"
        ]
        if chrono_cols:
            prompt += f"\nChronological columns (maintain time order per entity): {chrono_cols}\n"

        # Collect all programmatically-handled columns
        skip_cols = [
            col_name for col_name, role in relationships.columns.items()
            if role.role in ("auto", "derived")
        ]
        if skip_cols:
            prompt += f"\nDO NOT include these columns (computed programmatically): {skip_cols}\n"

        prompt += f"\nNumber of rows to generate: {count}\n"
        prompt += "Distribute events across MULTIPLE entities (not all for one entity).\n"

        # Add previous rows as context for continuity
        if all_rows:
            sample_rows = all_rows[-self.context_window:]
            prompt += f"\nPreviously generated rows ({len(all_rows)} total, showing last {len(sample_rows)} for continuity):\n"
            for row in sample_rows:
                prompt += json.dumps(row) + "\n"
            prompt += "\nContinue from where these left off. Maintain chronological order.\n"

        return prompt

    def _generate_entity_batch(
        self,
        specs: List[ColumnSpec],
        schema_str: str,
        entities: List[dict],
        count: int,
        all_rows: List[dict],
        relationships: RelationshipSpec,
        batch_num: int
    ) -> List[dict]:
        """Generate one batch of entity-aware rows."""

        user_prompt = self._build_entity_batch_prompt(
            schema_str, entities,
            count, all_rows, relationships
        )

        print(f"\n=== USER PROMPT (ENTITY BATCH {batch_num}) ===")
        print(user_prompt)

        raw = self.llm.complete(
            ENTITY_BATCH_GENERATION_SYSTEM,
            user_prompt
        )

        print(f"\n=== RAW LLM OUTPUT (ENTITY BATCH {batch_num}) ===")
        print(raw)

        rows = self._parse_jsonl(raw)

        # Remove auto + derived columns if the LLM included them anyway
        skip_cols = [
            col_name for col_name, role in relationships.columns.items()
            if role.role in ("auto", "derived")
        ]
        for row in rows:
            for col in skip_cols:
                row.pop(col, None)

        return rows

    # =========================================================
    # Derived Column Computation (NEW)
    # =========================================================
    def _compute_derived_columns(
        self,
        rows: List[dict],
        relationships: RelationshipSpec
    ) -> None:
        """Compute all derived columns using Python (in-place)."""

        entity_col = relationships.entity_column

        for col_name, role in relationships.columns.items():
            if role.role != "derived" or not role.compute:
                continue

            compute = role.compute
            method = compute.get("method", "")

            if method == "running_sum":
                self._compute_running_sum(rows, col_name, compute, entity_col)
            elif method == "running_average":
                self._compute_running_average(rows, col_name, compute, entity_col)
            elif method == "running_count":
                self._compute_running_count(rows, col_name, compute, entity_col)
            elif method == "running_min":
                self._compute_running_min(rows, col_name, compute, entity_col)
            elif method == "running_max":
                self._compute_running_max(rows, col_name, compute, entity_col)
            else:
                print(f"  ⚠ Unknown compute method '{method}' for '{col_name}' — skipping")

    def _get_signed_value(self, row: dict, compute: dict) -> float:
        """Extract the numerical value with correct sign based on compute spec."""
        value_col = compute.get("value_column", "")
        sign_col = compute.get("sign_column")
        pos_values = set(compute.get("positive_values", []))
        neg_values = set(compute.get("negative_values", []))

        try:
            value = float(row.get(value_col, 0))
        except (ValueError, TypeError):
            value = 0.0

        if sign_col:
            sign_val = str(row.get(sign_col, ""))
            if sign_val in neg_values:
                value = -value
            # If not in pos_values and not in neg_values, default to positive

        return value

    def _compute_running_sum(
        self, rows: List[dict], col_name: str, compute: dict, entity_col: str | None
    ) -> None:
        """Cumulative sum grouped by entity. Handles +/- sign column."""
        initial = compute.get("initial_value", 0)
        running = {}  # entity → current total

        for row in rows:
            entity = row.get(entity_col, "__all__") if entity_col else "__all__"
            if entity not in running:
                running[entity] = initial

            signed_value = self._get_signed_value(row, compute)
            running[entity] += signed_value
            row[col_name] = round(running[entity], 2)

        print(f"  ✔ Computed '{col_name}' (running_sum) for {len(running)} entities")

    def _compute_running_average(
        self, rows: List[dict], col_name: str, compute: dict, entity_col: str | None
    ) -> None:
        """Running average grouped by entity."""
        value_col = compute.get("value_column", "")
        totals = {}   # entity → cumulative sum
        counts = {}   # entity → count

        for row in rows:
            entity = row.get(entity_col, "__all__") if entity_col else "__all__"
            if entity not in totals:
                totals[entity] = 0.0
                counts[entity] = 0

            try:
                value = float(row.get(value_col, 0))
            except (ValueError, TypeError):
                value = 0.0

            totals[entity] += value
            counts[entity] += 1
            row[col_name] = round(totals[entity] / counts[entity], 2)

        print(f"  ✔ Computed '{col_name}' (running_average) for {len(totals)} entities")

    def _compute_running_count(
        self, rows: List[dict], col_name: str, compute: dict, entity_col: str | None
    ) -> None:
        """Running count grouped by entity."""
        counts = {}  # entity → count

        for row in rows:
            entity = row.get(entity_col, "__all__") if entity_col else "__all__"
            counts[entity] = counts.get(entity, 0) + 1
            row[col_name] = counts[entity]

        print(f"  ✔ Computed '{col_name}' (running_count) for {len(counts)} entities")

    def _compute_running_min(
        self, rows: List[dict], col_name: str, compute: dict, entity_col: str | None
    ) -> None:
        """Running minimum grouped by entity."""
        value_col = compute.get("value_column", "")
        mins = {}  # entity → current min

        for row in rows:
            entity = row.get(entity_col, "__all__") if entity_col else "__all__"

            try:
                value = float(row.get(value_col, 0))
            except (ValueError, TypeError):
                value = 0.0

            if entity not in mins:
                mins[entity] = value
            else:
                mins[entity] = min(mins[entity], value)

            row[col_name] = round(mins[entity], 2)

        print(f"  ✔ Computed '{col_name}' (running_min) for {len(mins)} entities")

    def _compute_running_max(
        self, rows: List[dict], col_name: str, compute: dict, entity_col: str | None
    ) -> None:
        """Running maximum grouped by entity."""
        value_col = compute.get("value_column", "")
        maxs = {}  # entity → current max

        for row in rows:
            entity = row.get(entity_col, "__all__") if entity_col else "__all__"

            try:
                value = float(row.get(value_col, 0))
            except (ValueError, TypeError):
                value = 0.0

            if entity not in maxs:
                maxs[entity] = value
            else:
                maxs[entity] = max(maxs[entity], value)

            row[col_name] = round(maxs[entity], 2)

        print(f"  ✔ Computed '{col_name}' (running_max) for {len(maxs)} entities")

    # =========================================================
    # Post-processing
    # =========================================================
    def _post_process(
        self,
        rows: List[dict],
        relationships: RelationshipSpec
    ) -> List[dict]:
        """Post-process: sort → compute derived → add auto columns."""

        if not rows:
            return rows

        entity_col = relationships.entity_column
        chrono_cols = [
            col_name for col_name, role in relationships.columns.items()
            if role.role == "chronological"
        ]

        # 1. Sort by entity + first chronological column
        if entity_col and chrono_cols:
            try:
                rows.sort(key=lambda r: (
                    str(r.get(entity_col, "")),
                    str(r.get(chrono_cols[0], ""))
                ))
                print(f"\n  ✔ Sorted rows by [{entity_col}, {chrono_cols[0]}]")
            except Exception as e:
                print(f"  ⚠ Could not sort rows: {e}")

        # 2. Compute derived columns (MUST happen after sort so running calcs are correct)
        has_derived = any(r.role == "derived" for r in relationships.columns.values())
        if has_derived:
            self._compute_derived_columns(rows, relationships)

        # 3. Add auto columns (sequential IDs)
        for col_name, role in relationships.columns.items():
            if role.role == "auto":
                prefix = role.prefix or ""
                pad = role.pad or 6
                for i, row in enumerate(rows):
                    row[col_name] = f"{prefix}{str(i + 1).zfill(pad)}"
                print(f"  ✔ Generated auto column '{col_name}' ({prefix}000001 ... {prefix}{str(len(rows)).zfill(pad)})")

        return rows

    # =========================================================
    # Entity-mode Pipeline (NEW)
    # =========================================================
    def _generate_entity_mode(
        self,
        specs: List[ColumnSpec],
        relationships: RelationshipSpec
    ) -> pd.DataFrame:
        """Generate data using entity-aware pipeline."""

        print("\n" + "=" * 60)
        print("🔗 ENTITY MODE: Generating with relationship awareness")
        print("=" * 60)

        # 1. Generate entities
        entities = self._generate_entities(relationships)

        # 2. Build schema for event generation (excludes auto columns)
        schema_str = self._build_entity_schema_str(specs, relationships)
        derivation_rules = self._build_derivation_rules(relationships)

        print(f"\n=== ENTITY SCHEMA ===")
        print(schema_str)
        if derivation_rules:
            print(f"\n=== DERIVATION RULES ===")
            print(derivation_rules)

        # 3. Generate events in batches
        all_rows = []
        remaining = self.size
        batch_num = 0
        max_retries = 3
        consecutive_failures = 0

        while remaining > 0:
            batch_num += 1
            batch_count = min(self.batch_size, remaining)

            print(f"\n>>> Generating entity batch {batch_num} ({batch_count} rows requested, {remaining} remaining)...")

            rows = self._generate_entity_batch(
                specs, schema_str, entities,
                batch_count, all_rows, relationships, batch_num
            )

            if len(rows) == 0:
                consecutive_failures += 1
                print(f"  ⚠ Got 0 rows (attempt {consecutive_failures}/{max_retries})")
                if consecutive_failures >= max_retries:
                    print(f"\n  ✖ Stopped after {max_retries} consecutive empty batches.")
                    print(f"    Generated {len(all_rows)}/{self.size} rows.")
                    break
            else:
                consecutive_failures = 0

            all_rows.extend(rows)
            remaining = self.size - len(all_rows)

            print(f"  ✔ Got {len(rows)} rows → Total: {len(all_rows)}/{self.size}")

        # 4. Post-process (sort + auto columns)
        print("\n=== POST-PROCESSING ===")
        all_rows = self._post_process(all_rows, relationships)

        # 5. Trim to exact size
        all_rows = all_rows[:self.size]
        df = pd.DataFrame(all_rows)

        # 6. Reorder columns to match original user-specified order
        ordered_columns = [col for col in self.columns if col in df.columns]
        return df[ordered_columns]

    # =========================================================
    # Simple-mode Pipeline (existing logic, refactored)
    # =========================================================
    def _build_user_prompt(
        self,
        schema_str: str,
        count: int,
        all_rows: List[dict],
        used_values: dict
    ) -> str:

        prompt = f"""
Dataset context:
{self.context}

Column schema:
{schema_str}

Number of rows:
{count}
"""

        # Add previous rows as context for consistency
        if all_rows:
            sample_rows = all_rows[-self.context_window:]
            prompt += f"\nPreviously generated rows ({len(all_rows)} total, showing last {len(sample_rows)} for reference):\n"
            for row in sample_rows:
                prompt += json.dumps(row) + "\n"
            prompt += "\nContinue generating rows that match the same style, patterns, and distribution as above.\n"

        # Add uniqueness constraints if applicable
        if self.unique_columns and used_values:
            has_constraints = False
            for col in self.unique_columns:
                if col in used_values and used_values[col]:
                    has_constraints = True
                    break

            if has_constraints:
                prompt += "\nIMPORTANT — Uniqueness constraints:\n"
                prompt += "The following columns MUST have unique values. Do NOT reuse ANY of these:\n"
                for col in self.unique_columns:
                    if col in used_values and used_values[col]:
                        vals = list(used_values[col])
                        sample = vals if len(vals) <= 50 else vals[-50:]
                        prompt += f"- {col} (already used {len(used_values[col])}): {sample}\n"
                prompt += "\nGenerate ENTIRELY DIFFERENT values for the unique columns above.\n"
                prompt += "Non-unique columns CAN repeat from the previous rows — that's expected.\n"

        return prompt

    def _deduplicate_rows(
        self,
        rows: List[dict],
        used_values: dict
    ) -> List[dict]:
        """Remove rows that have duplicate values in unique columns."""
        if not self.unique_columns:
            return rows

        clean_rows = []
        for row in rows:
            is_dup = False
            for col in self.unique_columns:
                val = row.get(col)
                if val is not None and val in used_values.get(col, set()):
                    is_dup = True
                    break
            if not is_dup:
                clean_rows.append(row)
                for col in self.unique_columns:
                    val = row.get(col)
                    if val is not None:
                        used_values.setdefault(col, set()).add(val)

        return clean_rows

    def _generate_batch(
        self,
        specs: List[ColumnSpec],
        schema_str: str,
        count: int,
        all_rows: List[dict],
        used_values: dict,
        batch_num: int
    ) -> List[dict]:

        user_prompt = self._build_user_prompt(schema_str, count, all_rows, used_values)

        print(f"\n=== USER PROMPT (BATCH {batch_num}) ===")
        print(user_prompt)

        raw = self.llm.complete(
            FULL_DATA_GENERATION_SYSTEM,
            user_prompt
        )

        print(f"\n=== RAW LLM OUTPUT (BATCH {batch_num}) ===")
        print(raw)

        rows = self._parse_jsonl(raw)

        # Deduplicate against previously generated rows
        rows = self._deduplicate_rows(rows, used_values)

        return rows

    def _generate_simple_mode(self, specs: List[ColumnSpec]) -> pd.DataFrame:
        """Generate data using simple batch pipeline (no entity relationships)."""

        print("\n" + "=" * 60)
        print("📋 SIMPLE MODE: Generating without entity relationships")
        print("=" * 60)

        schema_str = self._build_schema_str(specs)

        all_rows = []
        used_values = {col: set() for col in self.unique_columns}
        remaining = self.size
        batch_num = 0
        max_retries = 3
        consecutive_failures = 0

        while remaining > 0:
            batch_num += 1
            buffer = int(self.batch_size * 0.2) if self.unique_columns else 0
            batch_count = min(self.batch_size + buffer, remaining + buffer)

            print(f"\n>>> Generating batch {batch_num} ({batch_count} rows requested, {remaining} remaining)...")

            rows = self._generate_batch(
                specs, schema_str, batch_count, all_rows, used_values, batch_num
            )

            if len(rows) == 0:
                consecutive_failures += 1
                print(f"  ⚠ Got 0 new rows (attempt {consecutive_failures}/{max_retries})")
                if consecutive_failures >= max_retries:
                    print(f"\n  ✖ Stopped after {max_retries} consecutive empty batches.")
                    print(f"    Generated {len(all_rows)}/{self.size} rows.")
                    print(f"    Tip: reduce unique_columns or increase batch_size.")
                    break
            else:
                consecutive_failures = 0

            all_rows.extend(rows)
            remaining = self.size - len(all_rows)

            print(f"  ✔ Got {len(rows)} rows → Total: {len(all_rows)}/{self.size}")

        all_rows = all_rows[:self.size]
        df = pd.DataFrame(all_rows)

        ordered_columns = [col.name for col in specs if col.name in df.columns]
        return df[ordered_columns]

    # =========================================================
    # Main Entry Point
    # =========================================================
    def generate(self) -> pd.DataFrame:
        """Generate the synthetic dataset."""

        print("\n" + "=" * 60)
        print(f"🚀 SYNTHETICA — Generating {self.size} rows")
        print(f"   Context: {self.context}")
        print(f"   Columns: {self.columns}")
        print(f"   Batch size: {self.batch_size}")
        print("=" * 60)

        # Phase 1: Classify columns
        specs = self._classify_columns()

        # Phase 2: Infer numerical bounds
        specs = self._infer_numerical_bounds(specs)

        # Phase 3: Infer relationships
        relationships = self._infer_relationships(specs)

        # Phase 4: Generate based on detected mode
        if relationships.entity_column:
            return self._generate_entity_mode(specs, relationships)
        else:
            return self._generate_simple_mode(specs)
