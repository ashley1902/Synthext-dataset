"""
synthext — LLM-powered synthetic dataset generator.

Usage:
    from synthetica import Synthext

    df = Synthext(
        context="Retail banking transaction records",
        columns=["transaction_id", "customer_name", "amount", "balance"],
        size=100
    ).generate()
"""

from synthetica.generator import DatasetGenerator
import pandas as pd
from typing import List, Optional


class Synthext:

    def __init__(
        self,
        context: str,
        columns: List[str],
        size: int,
        api_key: Optional[str] = None
    ):
        self._generator = DatasetGenerator(
            context=context,
            columns=columns,
            size=size,
            openai_api_key=api_key,
            batch_size=50,
            context_window=20
        )

    def generate(self) -> pd.DataFrame:
        """Generate the synthetic dataset and return as a pandas DataFrame."""
        return self._generator.generate()


def generate(
    context: str,
    columns: List[str],
    size: int,
    api_key: Optional[str] = None
) -> pd.DataFrame:

    return Synthext(
        context=context,
        columns=columns,
        size=size,
        api_key=api_key
    ).generate()
