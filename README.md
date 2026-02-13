# synthext

**LLM-powered synthetic dataset generator** — LLM based synthetic dataset generator: context aware, relationship preserving and domain agnostic.

Give it a context, column names, and the required rows.  
It figures out the rest — types, bounds, relationships, entities, derived columns and returns a clean pandas DataFrame in the end.

## What This Library Solves

Generating synthetic data that actually makes sense is hard.

Traditional tools produce random noise - names that don't match IDs, balances that don't add up, dates are out of order, and columns with no logical relationship to each other.

`synthext` solves this by using LLM intelligence to:

- Understand your domain from a single context string
- Automatically classify columns as numerical or text
- Infer realistic bounds for numerical columns
- Detect entity relationships (e.g. same customer → same account number)
- Identify derived columns (e.g. running balance from transactions)
- Generate coherent, multi-batch data with chronological ordering
- Compute derived columns with exact Python arithmetic — no LLM math errors

## Key Features

- Single context string — no schema definitions needed
- Automatic column classification (numerical vs text, if numerical - integer vs float)
- ER aware generation (grouped, linked, chronological)
- Derived column computation (running sums, averages, counts, min, max)
- Auto-increment columns (sequential IDs with prefix/padding)
- Batch generation with context continuity to avoid errors and hallucinations
- Domain-agnostic — works for any sort of datasets (like medial, finance, sales etc.)
- Returns pandas DataFrame
- Powered by OpenAI GPT-4o-mini 

## Installation

```bash
pip install synthext
```

## Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass it directly:

```python
df = Synthext(context="...", columns=[...], size=100, api_key="sk-...").generate()
```

## Quick Start

```python
from synthetica import Synthext

df = Synthext(
    context="Retail banking transaction records for a mid-sized private bank",
    columns=[
        "transaction_id",
        "account_number",
        "customer_name",
        "transaction_date",
        "transaction_type",
        "merchant_name",
        "transaction_amount",
        "account_balance_after_transaction"
    ],
    size=500
).generate()

print(df.head(10))
df.to_csv("banking_data.csv", index=False)
```

**Output:**

| transaction_id | account_number | customer_name | transaction_date | transaction_type | merchant_name | transaction_amount | account_balance_after_transaction |
|---|---|---|---|---|---|---|---|
| TXN000001 | ACCT001 | Alice Johnson | 2023-10-01 | Deposit | Bank Transfer | 5000.0 | 5000.0 |
| TXN000002 | ACCT001 | Alice Johnson | 2023-10-29 | Withdrawal | ATM | 1000.0 | 4000.0 |
| TXN000003 | ACCT001 | Alice Johnson | 2023-11-26 | Transfer | Online Banking | 150.0 | 3850.0 |
| TXN000004 | ACCT002 | Brian Smith | 2023-10-02 | Deposit | Direct Deposit | 2500.0 | 2500.0 |
| TXN000005 | ACCT002 | Brian Smith | 2023-10-30 | Withdrawal | ATM | 800.0 | 1700.0 |

Notice:
- `ACCT001` always maps to `Alice Johnson` (entity + linked)
- `transaction_id` is auto-sequential (`TXN000001`, `TXN000002`, ...)
- `account_balance_after_transaction` is a mathematically correct running sum
- Dates are chronologically ordered per customer

## Functional API

For one-liner usage:

```python
from synthetica import generate

df = generate(
    context="E-commerce order records for an online electronics store",
    columns=["order_id", "customer_email", "customer_name", "order_date",
             "product_name", "quantity", "unit_price", "payment_method"],
    size=200
)
```

## How It Works

The library runs a multi-stage pipeline:

```
Context + Columns
       │
       ▼
┌──────────────────────────┐
│ 1. Column Classification │  LLM classifies each column as numerical or text
│    (numerical vs text)   │  with subtype (integer/float)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2. Numerical Bounds      │  LLM infers realistic min/max for each
│    Inference             │  numerical column based on domain context
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 3. Relationship          │  LLM detects column roles:
│    Inference             │  entity, linked, derived, auto,
│                          │  chronological, independent
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 4. Entity Generation     │  LLM creates unique entity master data
│                          │  (e.g. 30 customers with account numbers)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 5. Batch Event           │  LLM generates event rows in batches of 50
│    Generation            │  with context continuity between batches
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 6. Post-Processing       │  Python sorts by entity + chronological,
│                          │  computes derived columns with exact math,
│                          │  adds auto-increment IDs
└──────────┬───────────────┘
           ▼
      pandas DataFrame
```

## Column Roles (Auto-Detected)

The library automatically identifies these roles — no configuration needed:

| Role | Description | Example |
|---|---|---|
| **entity** | Primary grouping column — the main repeating subject | `customer_id`, `store_name`, `patient_id` |
| **linked** | Fixed value per entity — always the same for a given entity | `customer_name` linked to `customer_id` |
| **chronological** | Time-ordered column — maintains progression per entity | `transaction_date`, `visit_date` |
| **derived** | Computed from other columns — calculated by Python, not LLM | `running_balance`, `cumulative_spend` |
| **auto** | Sequential ID — generated programmatically with prefix + padding | `transaction_id` → `TXN000001` |
| **independent** | No special relationship — generated freely by LLM | `merchant_name`, `product_category` |

## Derived Column Computation

Derived columns are computed by Python (not the LLM) to ensure mathematical accuracy.

Supported methods:

| Method | Description | Example |
|---|---|---|
| `running_sum` | Cumulative sum per entity, with optional +/- sign logic | Account balance after each transaction |
| `running_average` | Running mean per entity | Average order value over time |
| `running_count` | Running count per entity | Total visits per patient |
| `running_min` | Running minimum per entity | Lowest score achieved |
| `running_max` | Running maximum per entity | Highest revenue recorded |

## Examples

### Banking Transactions

```python
from synthetica import Synthext

df = Synthext(
    context="Retail banking transaction records for a mid-sized private bank",
    columns=[
        "transaction_id",
        "account_number",
        "customer_name",
        "transaction_date",
        "transaction_type",
        "merchant_name",
        "transaction_amount",
        "account_balance_after_transaction",
        "is_fraud_suspected"
    ],
    size=100
).generate()
```

**What the library auto-detects:**
- `account_number` → entity
- `customer_name` → linked to `account_number`
- `transaction_date` → chronological
- `transaction_id` → auto (`TXN000001`, `TXN000002`, ...)
- `account_balance_after_transaction` → derived (running sum of `transaction_amount`, +Deposit / -Withdrawal)

### E-commerce Orders

```python
from synthetica import Synthext

df = Synthext(
    context="E-commerce order records for an online electronics and accessories store",
    columns=[
        "order_id",
        "customer_email",
        "customer_name",
        "order_date",
        "product_category",
        "product_name",
        "quantity",
        "unit_price",
        "payment_method",
        "shipping_city",
        "cumulative_customer_spend"
    ],
    size=500
).generate()
```

**What the library auto-detects:**
- `customer_email` → entity
- `customer_name` → linked to `customer_email`
- `order_date` → chronological
- `order_id` → auto
- `cumulative_customer_spend` → derived (running sum of order value per customer)

### Healthcare Patient Records

```python
from synthetica import Synthext

df = Synthext(
    context="Patient visit records for a multi-specialty hospital",
    columns=[
        "visit_id",
        "patient_id",
        "patient_name",
        "visit_date",
        "department",
        "diagnosis",
        "doctor_name",
        "bill_amount",
        "total_patient_spend"
    ],
    size=300
).generate()
```

**What the library auto-detects:**
- `patient_id` → entity
- `patient_name` → linked to `patient_id`
- `visit_date` → chronological
- `visit_id` → auto
- `total_patient_spend` → derived (running sum of `bill_amount` per patient)

### Sales Pipeline (CRM)

```python
from synthetica import Synthext

df = Synthext(
    context="B2B sales pipeline data for a SaaS company",
    columns=[
        "deal_id",
        "company_name",
        "contact_person",
        "stage",
        "deal_value",
        "close_date",
        "sales_rep"
    ],
    size=200
).generate()
```

### Simple Flat Dataset (No Entities)

Not every dataset has entity relationships. The library handles flat datasets too:

```python
from synthetica import Synthext

df = Synthext(
    context="Used car listings from an online marketplace",
    columns=[
        "listing_id",
        "car_brand",
        "model",
        "year",
        "mileage_km",
        "fuel_type",
        "transmission",
        "price_usd",
        "seller_city"
    ],
    size=500
).generate()
```

When no entity relationships are detected, the library falls back to a simpler generation mode — still batched and context-aware, but without entity grouping or derived columns.

## Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `context` | `str` | Yes | — | Domain description for the dataset |
| `columns` | `list[str]` | Yes | — | Column names |
| `size` | `int` | Yes | — | Number of rows to generate |
| `api_key` | `str` | No | `OPENAI_API_KEY` env var | OpenAI API key |

## Output

Returns a standard `pandas.DataFrame` with:

- All requested columns in the original order
- Correct data types (numerical columns as float/int, text as string)
- Entity consistency across all rows
- Mathematically correct derived columns
- Sequential auto-increment IDs

## Requirements

- Python >= 3.9
- OpenAI API key 

## Contributing

Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.

Please ensure tests and documentation are updated where applicable.

## License

MIT License
