from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.ingestion.service import IngestionService


def main() -> None:
    settings = get_settings()
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
    ingestion = IngestionService(db_path=settings.sqlite_path, es_url=settings.es_url, es_index=settings.es_index)
    results = ingestion.seed_sample_documents(sample_dir)
    if not results:
        print(f"No sample documents found in {sample_dir}")
        return
    print("Seeded sample corpus:")
    for line in results:
        print(f"- {line}")


if __name__ == "__main__":
    main()
