from openvox.stt.engine.local_agreement import LocalAgreement

def test_first_update_commits_nothing():
    la = LocalAgreement()
    committed, volatile = la.update(["the", "quick"])
    assert committed == []
    assert volatile == ["the", "quick"]

def test_common_prefix_becomes_committed():
    la = LocalAgreement()
    la.update(["the", "quick", "brown"])
    committed, volatile = la.update(["the", "quick", "brownish", "fox"])
    assert committed == ["the", "quick"]
    assert volatile == ["brownish", "fox"]

def test_committed_is_monotonic():
    la = LocalAgreement()
    la.update(["the", "quick", "brown"])
    la.update(["the", "quick", "brown", "fox"])   # commits the, quick, brown
    committed, _ = la.update(["the", "quick", "brown", "foxes"])
    assert committed == ["the", "quick", "brown"]

def test_finalize_returns_last_hypothesis_and_resets():
    la = LocalAgreement()
    la.update(["hello", "world"])
    la.update(["hello", "world"])
    final = la.finalize()
    assert final == ["hello", "world"]
    committed, volatile = la.update(["new", "start"])
    assert committed == []
    assert volatile == ["new", "start"]
