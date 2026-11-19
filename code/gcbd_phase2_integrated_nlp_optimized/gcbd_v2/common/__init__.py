from .geometry import energy_measure, swat_loss, spatial_grid_support, token_offset_support
from .cache import GeometryCacheWriter, GeometryCacheReader
from .fingerprints import canonical_hash, sha256_file, file_identity
from .selection import CalibrationBuffer, cosine_frontier, select_gcbd, matched_rate_select, entropy_from_logits
from .transport import semantic_cost, entropic_transport_loss
from .accounting import TeacherLedger
from .extraction import extract_geometry_cache
from .metrics import linear_cka
from .class_embeddings import encode_class_texts, save_class_embeddings
from .statistics import paired_seed_analysis, noninferiority
from .sampling import ShardAwareBatchSampler

__all__ = [
    "energy_measure", "swat_loss", "spatial_grid_support", "token_offset_support",
    "GeometryCacheWriter", "GeometryCacheReader", "canonical_hash", "sha256_file", "file_identity",
    "CalibrationBuffer", "cosine_frontier", "select_gcbd", "matched_rate_select", "entropy_from_logits",
    "semantic_cost", "entropic_transport_loss", "TeacherLedger", "extract_geometry_cache", "linear_cka",
    "encode_class_texts", "save_class_embeddings", "paired_seed_analysis", "noninferiority",
    "ShardAwareBatchSampler",
]
