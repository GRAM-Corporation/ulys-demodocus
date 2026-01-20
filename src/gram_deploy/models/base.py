"""Base model class with common functionality for all GRAM models."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
import json

from pydantic import BaseModel, ConfigDict


T = TypeVar("T", bound="GRAMModel")


class GRAMModel(BaseModel):
    """Base model class with JSON serialization support.

    All GRAM models inherit from this class to get consistent
    serialization/deserialization behavior.
    """

    model_config = ConfigDict(
        # Use enum values in serialization
        use_enum_values=True,
        # Validate field assignments
        validate_assignment=True,
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize model to JSON string.

        Args:
            indent: Indentation level for pretty printing (default: 2)

        Returns:
            JSON string representation of the model
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to dictionary.

        Returns:
            Dictionary representation of the model
        """
        return self.model_dump()

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Deserialize model from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            Model instance
        """
        return cls.model_validate_json(json_str)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Deserialize model from dictionary.

        Args:
            data: Dictionary to parse

        Returns:
            Model instance
        """
        return cls.model_validate(data)

    @classmethod
    def load_from_file(cls: type[T], file_path: str | Path) -> T:
        """Load model from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Model instance
        """
        path = Path(file_path)
        return cls.from_json(path.read_text())

    def save_to_file(self, file_path: str | Path, indent: int = 2) -> None:
        """Save model to a JSON file.

        Args:
            file_path: Path to save the JSON file
            indent: Indentation level for pretty printing (default: 2)
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent))


def utc_now() -> datetime:
    """Get current time in UTC with timezone awareness.

    Returns:
        Current datetime with UTC timezone
    """
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime has UTC timezone.

    Args:
        dt: Datetime to check/convert

    Returns:
        Datetime with UTC timezone, or None if input is None
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
