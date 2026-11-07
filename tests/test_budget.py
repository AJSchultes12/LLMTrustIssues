import pytest

from llmtrust.budget import Budget, BudgetExceeded, estimate_tokens
from llmtrust.models import Arm


def _arm(**kw):
    base = dict(name="a", provider="openai_compat", model="m", base_url="http://x",
                api_key_env=None, temperature=0.0, usd_per_mtok_in=1.0,
                usd_per_mtok_out=5.0, pool="other")
    base.update(kw)
    return Arm(**base)


def test_budget_blocks_before_cap(tmp_path):
    b = Budget({"claude": 8.0}, tmp_path / "b.json")
    b.record("claude", 7.9)
    with pytest.raises(BudgetExceeded):
        b.check("claude", 0.2)
    b.check("claude", 0.05)


def test_budget_equal_to_cap_allowed(tmp_path):
    b = Budget({"claude": 8.0}, tmp_path / "b.json")
    b.record("claude", 7.5)
    b.check("claude", 0.5)


def test_budget_persists_across_instances(tmp_path):
    p = tmp_path / "b.json"
    Budget({"claude": 8.0}, p).record("claude", 3.0)
    b2 = Budget({"claude": 8.0}, p)
    assert b2.spent("claude") == pytest.approx(3.0)
    with pytest.raises(BudgetExceeded):
        b2.check("claude", 5.5)


def test_budget_uncapped_pool_never_raises(tmp_path):
    b = Budget({"claude": 8.0}, tmp_path / "b.json")
    b.record("free", 1000.0)
    b.check("free", 1000.0)


def test_budget_leaves_no_temp_files(tmp_path):
    b = Budget({"claude": 8.0}, tmp_path / "b.json")
    b.record("claude", 1.0)
    assert [p.name for p in tmp_path.iterdir()] == ["b.json"]


def test_estimate_usd_uses_per_million_prices(tmp_path):
    b = Budget({"other": 20.0}, tmp_path / "b.json")
    arm = _arm(usd_per_mtok_in=2.0, usd_per_mtok_out=10.0)
    assert b.estimate_usd(arm, 1_000_000, 500_000) == pytest.approx(7.0)


def test_estimate_usd_free_pool_is_zero(tmp_path):
    b = Budget({}, tmp_path / "b.json")
    arm = _arm(pool="free", usd_per_mtok_in=None, usd_per_mtok_out=None)
    assert b.estimate_usd(arm, 10**9, 10**9) == 0.0


def test_estimate_usd_null_price_on_paid_arm_raises(tmp_path):
    b = Budget({"other": 20.0}, tmp_path / "b.json")
    arm = _arm(usd_per_mtok_in=None, usd_per_mtok_out=None)
    with pytest.raises(ValueError, match="price"):
        b.estimate_usd(arm, 1000, 1000)


def test_estimate_tokens_ceil_div_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
