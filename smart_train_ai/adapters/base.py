from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from smart_train_ai.schema import SensorRecord


class DataSource(ABC):
    """Abstract interface decoupling ML pipeline from data ingestion sources."""

    @abstractmethod
    def fetch_latest_record(self) -> SensorRecord:
        """Fetches single latest SensorRecord."""
        pass

    @abstractmethod
    def fetch_window_df(self, duration_seconds: float = 5.0) -> pd.DataFrame:
        """Fetches telemetry DataFrame for requested duration."""
        pass
