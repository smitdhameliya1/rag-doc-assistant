from app.ingest import chunk_text


def test_chunks_respect_size():
    text = "word " * 1000
    chunks = chunk_text(text, size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_overlap_shared_between_chunks():
    text = "".join(str(i % 10) for i in range(500))
    chunks = chunk_text(text, size=100, overlap=20)
    assert chunks[0][-20:] == chunks[1][:20]


def test_short_text_dropped():
    assert chunk_text("tiny") == []
