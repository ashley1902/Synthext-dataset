from synthetica.generator import DatasetGenerator

gen = DatasetGenerator(
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
    size=100,
    batch_size=25,
    context_window=20
)

df = gen.generate()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(df.describe(include="all"))
print("\nFirst 20 rows:")
print(df.head(20).to_string())

df.to_csv("output.csv", index=False)
print("\nSaved to output.csv")
