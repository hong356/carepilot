from __future__ import annotations

from app.rag.tokenizer import tokenize


def test_tokenize_chinese_bigrams():
    tokens = tokenize("七天无理由")
    assert "七天" in tokens
    assert "无理由" not in tokens  # bigram 切分：无理 / 理由
    assert "理由" in tokens


def test_shipping_insurance_policy(policies):
    result = policies.search("运费险怎么赔", 3)
    assert result["results"]
    assert result["results"][0]["policy_id"] == "P-LOG-002"


def test_fresh_food_policy(policies):
    result = policies.search("生鲜为什么不支持七天无理由", 3)
    ids = {item["policy_id"] for item in result["results"]}
    assert "P-RET-002" in ids


def test_large_refund_policy(policies):
    result = policies.search("超过500元退款怎么处理", 3)
    assert result["results"][0]["policy_id"] == "P-REF-003"


def test_mobile_activation_policy(policies):
    result = policies.search("手机激活了还能退吗", 3)
    assert result["results"][0]["policy_id"] == "P-RET-002"


def test_results_are_citable(policies):
    result = policies.search("价保期限", 4)
    assert result["policy_ids"]
    for item in result["results"]:
        assert item["policy_id"].startswith("P-")
        assert item["content"]
