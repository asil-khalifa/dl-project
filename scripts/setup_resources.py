"""One-time download of the spaCy model and NLTK data used by StudyQG."""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
import nltk  # noqa: E402

for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)
print("resources ready")
