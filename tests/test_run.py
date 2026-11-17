import pytest

from bsd.run import check_sha

SHA = "c22cc48bd24841d9dffb61f04aa1df26fac3647d17bb8a676c9f9d5d840f9eee"


def test_check_sha_match_passes():
    check_sha(SHA, SHA)
    check_sha(SHA, "  " + SHA.upper() + "\n")  # case and surrounding whitespace are ignored


def test_check_sha_mismatch_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        check_sha(SHA, "0" * 64)
    assert exc.value.code not in (0, None)
    assert "0" * 64 in str(exc.value.code) and SHA in str(exc.value.code)
