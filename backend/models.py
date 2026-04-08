from pydantic import BaseModel
from typing import Optional, List


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int


class AskRequest(BaseModel):
    doc_id: str
    question: str


class SourceChunk(BaseModel):
    text: str
    page_number: int
    similarity: float
    chunk_type: str  # "narrative" | "kv_block" | "table_row"


class AskResponse(BaseModel):
    answer: str
    confidence: float
    source_chunks: List[SourceChunk]
    guardrail_triggered: bool


class ShipperConsignee(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None


class ExtractionResult(BaseModel):
    shipment_id: Optional[str] = None
    shipper: Optional[ShipperConsignee] = None
    consignee: Optional[ShipperConsignee] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[float] = None
    currency: Optional[str] = None
    weight: Optional[float] = None
    carrier_name: Optional[str] = None


class ExtractRequest(BaseModel):
    doc_id: str


class ExtractResponse(BaseModel):
    doc_id: str
    data: ExtractionResult


# -- Collection management --------------------------------------------------

class CollectionChunkInfo(BaseModel):
    chunk_index: int
    chunk_type: str
    page_number: int
    text_preview: str


class CollectionInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    upload_timestamp: str
    chunks: List[CollectionChunkInfo]


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    upload_timestamp: Optional[str] = None
