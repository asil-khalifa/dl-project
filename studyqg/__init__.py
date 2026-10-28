"""StudyQG: automatic question generation from study material (CSPE72 Team 3)."""
from .config import DEFAULT_QG_MODEL, OUR_QG_MODEL, QA_MODEL, SMALL_QG_MODEL, GenConfig
from .pipeline import GenerationResult, Question, StudyQG

__version__ = "1.0.0"
__all__ = ["StudyQG", "GenConfig", "GenerationResult", "Question", "DEFAULT_QG_MODEL", "SMALL_QG_MODEL",
           "OUR_QG_MODEL", "QA_MODEL"]
