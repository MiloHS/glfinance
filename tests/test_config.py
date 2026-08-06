from pathlib import Path

from glfinance.config import load_config


def test_repository_config_is_valid() -> None:
    config = load_config(Path("configs/portfolio_walk_forward.yaml"))
    assert config["portfolio"]["primary_method"] == "GraphLW"
    assert config["data"]["universe_sizes"] == [12, 24, 36]
    assert len(config["data"]["tickers"]) == 36

