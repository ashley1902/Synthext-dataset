from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any


class ColumnSpec(BaseModel):
    name: str
    type: Literal["numerical", "text"]
    subtype: Optional[Literal["integer", "float"]] = None
    min: Optional[float] = None
    max: Optional[float] = None


class ColumnRole(BaseModel):
    role: Literal["entity", "linked", "derived", "auto", "chronological", "independent"]
    linked_to: Optional[str] = None
    depends_on: Optional[List[str]] = None
    logic: Optional[str] = None
    compute: Optional[Dict[str, Any]] = None  # structured spec for Python computation
    prefix: Optional[str] = None
    pad: Optional[int] = None


class RelationshipSpec(BaseModel):
    entity_column: Optional[str] = None
    columns: Dict[str, ColumnRole] = {}
