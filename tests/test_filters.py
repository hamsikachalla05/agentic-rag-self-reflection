from ingestion.chunker import Chunk


def make_chunk(**kwargs) -> Chunk:
    base = dict(
        chunk_id="doc::p1::c0::aaaa",
        document_id="northstar-financial-inc_2023_annual-report",
        page=3,
        pages=[3],
        company="Northstar Financial Inc",
        year=2023,
        filing_type="annual report",
        source_filename="Northstar-Financial-Inc_2023_annual-report.pdf",
        text="Total revenues were $12.40 billion.",
        token_count=12,
        extra_metadata={},
    )
    base.update(kwargs)
    return Chunk(**base)


def test_filter_company_year():
    chunk = make_chunk()
    assert chunk.matches_filters({"company": "Northstar Financial Inc", "year": 2023})
    assert not chunk.matches_filters({"year": 2022})
    assert not chunk.matches_filters({"company": "CedarTech PLC"})


def test_filter_document_id_and_filing_type():
    chunk = make_chunk()
    assert chunk.matches_filters({"document_id": chunk.document_id})
    assert chunk.matches_filters({"filing_type": "annual report"})
    assert not chunk.matches_filters({"filing_type": "10-Q"})


def test_empty_filters_match_all():
    chunk = make_chunk()
    assert chunk.matches_filters(None)
    assert chunk.matches_filters({})
