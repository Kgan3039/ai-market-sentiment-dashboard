"""Phase 0 ingestion and persistence package."""

from .errors import (
    Phase0Error,
    Phase0IntegrityError,
    Phase0MigrationError,
    Phase0RunContextError,
    Phase0ValidationError,
    StageKeyError,
    UnsupportedTickerError,
)
from .models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    ProviderConflictRecord,
    ReconciliationReport,
    SemanticMergeRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from .redaction import contains_credential, redact_secrets, redact_text
from .scalars import (
    require_safe_identifier_scalar,
    sanitize_diagnostic_scalar,
    validate_safe_identifier_scalar,
)
from .repository import (
    Phase0Admin,
    Phase0Repository,
    StageRunContext,
    serialize_operational_metadata,
    serialize_raw_evidence,
)
from .tickers import SUPPORTED_TICKERS, TICKER_UNIVERSE, normalize_ticker

__all__ = [
    "ExcludedStoryRecord",
    "OtherCoverageRecord",
    "Phase0Error",
    "Phase0IntegrityError",
    "Phase0Admin",
    "Phase0MigrationError",
    "Phase0Repository",
    "Phase0RunContextError",
    "Phase0ValidationError",
    "ProviderConflictRecord",
    "ReconciliationReport",
    "SUPPORTED_TICKERS",
    "SemanticMergeRecord",
    "StageKeyError",
    "StageRunContext",
    "StoryMemberRecord",
    "StoryRecord",
    "TICKER_UNIVERSE",
    "ThemeRecord",
    "ThemeSetRecord",
    "UnsupportedTickerError",
    "contains_credential",
    "normalize_ticker",
    "redact_secrets",
    "redact_text",
    "require_safe_identifier_scalar",
    "sanitize_diagnostic_scalar",
    "serialize_operational_metadata",
    "serialize_raw_evidence",
    "validate_safe_identifier_scalar",
]
