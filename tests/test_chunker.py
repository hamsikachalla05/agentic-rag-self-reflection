from ingestion.chunker import chunk_document
from ingestion.pdf_parser import ParsedDocument, ParsedPage


def test_chunk_ids_unique_and_keep_metadata():
    pages = [
        ParsedPage(page=1, text="alpha " * 400),
        ParsedPage(page=2, text="beta " * 400),
    ]
    doc = ParsedDocument(
        document_id="demo-2023",
        source_filename="demo.pdf",
        source_path="/tmp/demo.pdf",
        company="Demo Co",
        year=2023,
        filing_type="annual report",
        pages=pages,
    )
    chunks = chunk_document(doc, chunk_size_tokens=64, chunk_overlap_tokens=8)
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for chunk in chunks:
        assert chunk.document_id == "demo-2023"
        assert chunk.company == "Demo Co"
        assert chunk.year == 2023
        assert chunk.source_filename == "demo.pdf"
        assert chunk.page >= 1
        assert chunk.text
        assert chunk.token_count <= 64
