from bsd.fetch_sample import SAMPLE_SHA, sample_urls


def test_urls_pinned():
    urls = sample_urls()
    assert len(urls) == 9
    assert "Sample_Game_3/Sample_Game_3_tracking.txt" in urls
    assert all(SAMPLE_SHA in u for u in urls.values())
