from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gdp_engine import (
    GDPAnalyzer, RegimeDetector, StatisticsEngine,
    BacktestEngine, DataPoint, EnhancedBacktestResult,
)


class MacroAnalytics:
    @staticmethod
    def load_history(path: str = "data/analysis_log.csv") -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        return df

    @staticmethod
    def correlation_matrix(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        if columns is None:
            cols = ["cpi", "core_gdp", "nfp", "dxy", "treasury_yield",
                    "treasury_3m_change", "unemployment", "consumer_sentiment", "pmi"]
        else:
            cols = columns
        available = [c for c in cols if c in df.columns and df[c].notna().sum() > 5]
        return df[available].corr(method="pearson")

    @staticmethod
    def rolling_correlation(df: pd.DataFrame, col_a: str, col_b: str,
                            window: int = 60) -> pd.Series:
        return df[col_a].rolling(window).corr(df[col_b])

    @staticmethod
    def signal_accuracy(df: pd.DataFrame) -> dict:
        if df.empty or "signal" not in df.columns:
            return {}
        total = len(df)
        counts = df["signal"].value_counts()
        return {
            "total_runs": total,
            "signal_distribution": counts.to_dict(),
            "buy_pct": round(counts.get("BUY", 0) / total * 100, 1) if total else 0,
            "sell_pct": round(counts.get("SELL", 0) / total * 100, 1) if total else 0,
            "hold_pct": round(counts.get("HOLD", 0) / total * 100, 1) if total else 0,
            "pending_pct": round(counts.get("PENDING", 0) / total * 100, 1) if total else 0,
            "strong_buy_pct": round(counts.get("STRONG_BUY", 0) / total * 100, 1) if total else 0,
            "strong_sell_pct": round(counts.get("STRONG_SELL", 0) / total * 100, 1) if total else 0,
        }

    @staticmethod
    def regime_breakdown(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        results = []
        for _, row in df.iterrows():
            try:
                detector = RegimeDetector(
                    cpi=float(row.get("cpi", 0)),
                    core_gdp=float(row.get("core_gdp", 0)),
                    nfp=float(row.get("nfp", 0)),
                    dxy=float(row.get("dxy", 0)),
                    treasury_yield=float(row.get("treasury_yield", 0)),
                    unemployment_rate=float(row.get("unemployment", 0)),
                    consumer_sentiment=float(row.get("consumer_sentiment", 0)),
                    yield_spread_10y2y=float(row.get("yield_spread_10y2y", 0)),
                )
                r = detector.analyze()
                results.append({
                    "timestamp": row.get("timestamp"),
                    "regime": r.regime,
                    "expansion_prob": r.expansion_prob,
                    "inflation_prob": r.inflation_prob,
                    "recession_prob": r.recession_prob,
                    "stagflation_prob": r.stagflation_prob,
                    "normal_prob": r.normal_prob,
                })
            except Exception:
                continue
        return pd.DataFrame(results)

    @staticmethod
    def regime_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
        regimes = MacroAnalytics.regime_breakdown(df)
        if regimes.empty or "regime" not in regimes.columns:
            return pd.DataFrame()

        labels = regimes["regime"].apply(lambda x: str(x).split(".")[1] if hasattr(x, "name") else str(x))
        transitions = []
        for i in range(1, len(labels)):
            transitions.append((labels.iloc[i - 1], labels.iloc[i]))

        tf = pd.DataFrame(transitions, columns=["from", "to"])
        matrix = tf.groupby(["from", "to"]).size().unstack(fill_value=0)
        matrix = matrix.div(matrix.sum(axis=1), axis=0) * 100
        return matrix

    @staticmethod
    def detect_anomalies(df: pd.DataFrame, columns: list[str] = None,
                         zscore_threshold: float = 2.5) -> pd.DataFrame:
        if columns is None:
            columns = ["cpi", "core_gdp", "nfp", "dxy", "treasury_yield",
                       "consumer_sentiment", "pmi"]
        available = [c for c in columns if c in df.columns and df[c].notna().sum() > 10]
        if not available:
            return pd.DataFrame()

        result = df[["timestamp"]].copy() if "timestamp" in df.columns else pd.DataFrame(index=df.index)
        for col in available:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            mean_v = col_data.mean()
            std_v = col_data.std()
            if std_v == 0:
                continue
            zscores = (df[col] - mean_v) / std_v
            result[f"{col}_zscore"] = zscores.round(2)
            result[f"{col}_anomaly"] = zscores.abs() > zscore_threshold

        anomaly_cols = [c for c in result.columns if c.endswith("_anomaly")]
        if anomaly_cols:
            result["total_anomalies"] = result[anomaly_cols].sum(axis=1)
        return result

    @staticmethod
    def indicator_momentum(df: pd.DataFrame, column: str, window: int = 3) -> pd.Series:
        if column not in df.columns:
            return pd.Series(dtype=float)
        return df[column].pct_change(periods=window) * 100

    @staticmethod
    def composite_health_score(df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        if "cpi" in df.columns:
            score += df["cpi"].apply(lambda x: -2 if x > 4.5 else -1 if x > 3 else 0 if x > 2 else 1)
        if "core_gdp" in df.columns:
            score += df["core_gdp"].apply(lambda x: 1 if x > 2.5 else 0 if x > 1 else -1)
        if "nfp" in df.columns:
            score += df["nfp"].apply(lambda x: 1 if x > 200000 else 0 if x > 100000 else -1)
        if "unemployment" in df.columns:
            score += df["unemployment"].apply(lambda x: -1 if x > 6 else 0 if x > 4 else 1)
        if "consumer_sentiment" in df.columns:
            score += df["consumer_sentiment"].apply(lambda x: 1 if x > 80 else 0 if x > 60 else -1)
        return score

    @staticmethod
    def sensitivity_analysis(base_data: dict, variable: str,
                             range_pct: float = 0.5, steps: int = 11) -> pd.DataFrame:
        base_val = base_data.get(variable, 0)
        if base_val == 0:
            return pd.DataFrame()

        results = []
        for multiplier in np.linspace(1 - range_pct, 1 + range_pct, steps):
            test_data = base_data.copy()
            test_data[variable] = base_val * multiplier

            try:
                analyzer = GDPAnalyzer(
                    cpi=test_data.get("cpi", 0),
                    core_gdp=test_data.get("core_gdp", 0),
                    nfp=test_data.get("nfp", 0),
                    dxy=test_data.get("dxy", 0),
                    treasury_3m_pct_change=test_data.get("treasury_3m_change", 0),
                    treasury_yield=test_data.get("treasury_yield", 0),
                    unemployment_rate=test_data.get("unemployment", 0),
                    consumer_sentiment=test_data.get("consumer_sentiment", 0),
                    yield_spread_10y2y=test_data.get("yield_spread_10y2y", 0),
                )
                results.append({
                    f"{variable}_value": round(base_val * multiplier, 2),
                    "signal": analyzer.signal,
                    "confidence": analyzer.confidence,
                    "ensemble_value": round(analyzer.get_ensemble_numeric_value(), 4),
                    "signal_strength": round(analyzer.get_signal_strength(), 4),
                    "dominant_model": analyzer.get_dominant_model(),
                    "model_agreement": round(analyzer.get_model_agreement(), 4),
                })
            except Exception as e:
                continue

        return pd.DataFrame(results)

    @staticmethod
    def full_report(base_data: dict) -> dict:
        analyzer = GDPAnalyzer(
            cpi=base_data.get("cpi", 0),
            core_gdp=base_data.get("core_gdp", 0),
            nfp=base_data.get("nfp", 0),
            dxy=base_data.get("dxy", 0),
            treasury_3m_pct_change=base_data.get("treasury_3m_change", 0),
            treasury_yield=base_data.get("treasury_yield", 0),
            unemployment_rate=base_data.get("unemployment", 0),
            consumer_sentiment=base_data.get("consumer_sentiment", 0),
            yield_spread_10y2y=base_data.get("yield_spread_10y2y", 0),
        )

        regime = analyzer.get_regime()
        votes = analyzer.get_model_votes()
        names = ["InflationVeto", "TaylorRule", "DollarStrength",
                 "BondSignal", "CreditSpread", "LaborMarket", "GlobalRisk"]

        return {
            "signal": analyzer.signal,
            "confidence": analyzer.confidence,
            "ensemble_value": round(analyzer.get_ensemble_numeric_value(), 4),
            "signal_strength": round(analyzer.get_signal_strength(), 4),
            "model_agreement": round(analyzer.get_model_agreement(), 4),
            "dominant_model": analyzer.get_dominant_model(),
            "reason": analyzer.reason,
            "methodology": analyzer.get_methodology_summary(),
            "regime": {
                "label": str(regime.regime).split(".")[1],
                "expansion_prob": round(regime.expansion_prob, 3),
                "inflation_prob": round(regime.inflation_prob, 3),
                "recession_prob": round(regime.recession_prob, 3),
                "stagflation_prob": round(regime.stagflation_prob, 3),
                "normal_prob": round(regime.normal_prob, 3),
            },
            "models": [
                {"name": names[i] if i < len(names) else f"Model {i}",
                 "signal": v.signal,
                 "confidence": v.confidence,
                 "value": round(v.numeric_value, 3),
                 "reason": v.reason}
                for i, v in enumerate(votes)
            ],
        }

    @staticmethod
    def run_backtest(api_key: str, years: int = 20) -> dict:
        from python.data_collector import prepare_backtest_data
        data_points, prices = prepare_backtest_data(api_key, years)
        engine = BacktestEngine(data_points, prices)

        strategy_result = engine.run_all_models()
        benchmark_result = engine.run_buy_and_hold()
        walkforward = engine.run_walk_forward(120, 60)
        risk_controlled = engine.run_with_risk_controls(8.0, 25.0)
        sized = engine.run_with_position_sizing()

        def _result_to_dict(r: EnhancedBacktestResult, label: str) -> dict:
            return {
                "label": label,
                "total_return_pct": round(r.total_return_pct, 2),
                "annualized_return_pct": round(r.annualized_return_pct, 2),
                "volatility_pct": round(r.volatility_pct, 2),
                "sharpe_ratio": round(r.sharpe_ratio, 3),
                "sortino_ratio": round(r.sortino_ratio, 3),
                "calmar_ratio": round(r.calmar_ratio, 3),
                "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                "var_95_pct": round(r.var_95_pct, 2),
                "cvar_95_pct": round(r.cvar_95_pct, 2),
                "ulcer_index": round(r.ulcer_index, 2),
                "win_rate_pct": round(r.win_rate_pct, 1),
                "profit_factor": round(r.profit_factor, 2),
                "total_trades": r.total_trades,
                "final_value": round(r.final_value, 2),
            }

        return {
            "strategy": _result_to_dict(strategy_result, "Ensemble Strategy"),
            "benchmark": _result_to_dict(benchmark_result, "Buy & Hold"),
            "walk_forward": _result_to_dict(walkforward, "Walk-Forward"),
            "risk_controlled": _result_to_dict(risk_controlled, "Risk-Controlled (SL/TP)"),
            "position_sized": _result_to_dict(sized, "Position-Sized"),
            "outperformance": round(
                strategy_result.total_return_pct - benchmark_result.total_return_pct, 2
            ),
        }