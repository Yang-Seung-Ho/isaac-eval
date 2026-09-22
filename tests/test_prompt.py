from isaac_eval.catalog import build_catalog_entry
from isaac_eval.gold import load_dataset
from isaac_eval.prompt import build_data_prompt, assert_prompt_clean
from isaac_eval.config import load_settings


def test_prompt_has_no_leaked_observation_values():
    cfg = load_settings()
    catalog = build_catalog_entry(cfg)
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
    prompt = build_data_prompt(catalog, "봄철 최고 발생 월과 정점은?")
    assert assert_prompt_clean(prompt, df) == []
    assert "df" in prompt
    assert "json.dumps" in prompt
    assert "int()" in prompt
