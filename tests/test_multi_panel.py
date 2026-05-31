"""De-hardcoding regression guard: a non-default panel must drive the pipeline."""
import os, sys, pickle, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

TECH = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META']


def _make_synthetic_pickle(path, tickers, n_days=80):
    """date -> {'price': {ticker: {close, open, high, low, volume, adjusted_close}}}."""
    rng = np.random.RandomState(0)
    dates = [f"2022-{1 + d // 28:02d}-{1 + d % 28:02d}" for d in range(n_days)]
    data = {}
    for di, date in enumerate(dates):
        row = {}
        for ti, t in enumerate(tickers):
            base = 100.0 + ti * 10 + di * 0.5
            row[t] = {'close': base, 'open': base, 'high': base * 1.01,
                      'low': base * 0.99, 'volume': 1e6 + ti * 1e5,
                      'adjusted_close': base}
        data[date] = {'price': row}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return dates


def test_env_uses_config_tickers():
    from portfolio_env import PortfolioEnv
    from regime_detector import detect_market_regime
    with tempfile.TemporaryDirectory() as d:
        pkl = os.path.join(d, 'tech.pkl')
        dates = _make_synthetic_pickle(pkl, TECH)
        config = {'data': {'tickers': TECH}, 'portfolio': {'default_lambda': 0.5}}
        env = PortfolioEnv(pkl, config, detect_regime_fn=detect_market_regime,
                           train_period=(dates[0], dates[-1]))
        assert env.tickers == TECH
        rs = env._get_raw_states_dict(20)
        assert set(rs.keys()) == set(TECH)
        regime = detect_market_regime(rs)
        assert len(regime) == 3
        assert env.state_dim > 0


def test_sector_exposure_uses_config_groups():
    from portfolio_features import build_portfolio_features
    raw = {t: np.abs(np.random.RandomState(1).randn(120) * 5 + 100) for t in TECH}
    fn = build_portfolio_features(
        [{'indicator': 'sector_exposure'}],
        tickers=TECH, growth=['AAPL', 'NVDA', 'GOOGL', 'META'], defensive=['MSFT'])
    weights = np.array([0.3, 0.1, 0.2, 0.2, 0.1, 0.1])
    out = fn(raw, current_weights=weights)
    assert np.allclose(out[:2], [0.8, 0.1])


if __name__ == '__main__':
    test_env_uses_config_tickers()
    test_sector_exposure_uses_config_groups()
    print("OK")
