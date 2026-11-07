import os
import subprocess

import pytest

from graphswap.repos import checkout

COMMIT = "0b401c76b6e80a4eecf3c690085b2553f6e261ca"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("GRAPHSWAP_NETWORK"), reason="set GRAPHSWAP_NETWORK=1")
def test_checkout_requests(tmp_path):
    wt = checkout("psf/requests", COMMIT, tmp_path)
    assert wt == tmp_path / "wt" / f"requests_{COMMIT}"
    assert (wt / "src/requests/cookies.py").exists()
    head = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head == COMMIT
    assert checkout("psf/requests", COMMIT, tmp_path) == wt  # reuse
