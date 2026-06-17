#!/usr/bin/env python3
"""GDP Analysis Engine — Backend REST API"""

import os
import sys
import subprocess
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, request


def _ensure_built():
    try:
        import gdp_engine
    except ImportError:
        print("  Building C++ engine...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                           capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        if r.returncode != 0:
            print(f"[ERROR] Build failed:\n{r.stderr}")
            sys.exit(1)


_ensure_built()
load_dotenv()

from python.data_collector import run_analysis, fetch_historical_data, FRED_SERIES, prepare_backtest_data
from python.analytics import MacroAnalytics
from gdp_engine import GDPAnalyzer, StatisticsEngine, BacktestEngine

app = Flask(__name__)


@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r


def _get_key():
    return os.environ.get("FRED_API_KEY", "")


# ── Health & Status ──────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    key = _get_key()
    return jsonify({
        "status": "ok",
        "api_key_set": bool(key),
        "engine": "gdp_engine v2.0",
        "models": [
            "InflationVeto", "TaylorRule", "DollarStrength",
            "BondSignal", "CreditSpread", "LaborMarket", "GlobalRisk"
        ],
        "total_models": 7,
        "endpoints": [
            "/api/health", "/api/data", "/api/analyze",
            "/api/history", "/api/history/indicator/<name>",
            "/api/correlation", "/api/regime/history",
            "/api/backtest", "/api/backtest/detailed",
            "/api/scenario", "/api/indicators",
            "/api/anomalies", "/api/health-score",
            "/api/regime/transitions", "/api/report"
        ],
    })


# ── Data ─────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    key = _get_key()
    return jsonify({
        "status": "ok",
        "api_key_set": bool(key),
        "engine": "gdp_engine v2.0",
    })


@app.route("/api/data")
def api_data():
    key = _get_key()
    if not key:
        return jsonify({"error": "No FRED_API_KEY set"}), 400
    result = run_analysis(key)
    result["timestamp"] = datetime.utcnow().isoformat()
    return jsonify(result)


# ── Full Analysis ────────────────────────────────────────────────────

@app.route("/api/analyze")
def api_analyze():
    key = _get_key()
    if not key:
        return jsonify({"error": "No FRED_API_KEY set"}), 400
    data = run_analysis(key)
    analytics = MacroAnalytics()
    report = analytics.full_report(data)
    report["timestamp"] = datetime.utcnow().isoformat()
    report["raw_data"] = {
        k: v for k, v in data.items()
        if k in ("cpi", "core_gdp", "nfp", "dxy", "treasury_yield",
                 "treasury_2y", "treasury_3m_change", "yield_spread_10y2y",
                 "unemployment", "consumer_sentiment", "pmi", "fed_funds")
    }
    return jsonify(report)


# ── History ──────────────────────────────────────────────────────────

@app.route("/api/history")
def api_history():
    df = MacroAnalytics.load_history()
    if df.empty:
        return jsonify([])
    return jsonify(df.tail(100).to_dict(orient="records"))


@app.route("/api/history/indicator/<name>")
def api_history_indicator(name):
    key = _get_key()
    if not key:
        return jsonify({"error": "No FRED_API_KEY set"}), 400
    if name not in FRED_SERIES:
        return jsonify({"error": f"Unknown indicator: {name}"}), 400
    try:
        from python.data_collector import MacroDataCollector
        collector = MacroDataCollector(key)
        series = collector._fetch_series(FRED_SERIES[name], name)
        if series.empty:
            return jsonify([])
        return jsonify({
            "indicator": name,
            "series_id": FRED_SERIES[name],
            "data": [
                {"date": str(k.date()), "value": round(float(v), 4)}
                for k, v in series.items()
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Correlation ──────────────────────────────────────────────────────

@app.route("/api/correlation")
def api_correlation():
    key = _get_key()
    if not key:
        return jsonify({"error": "No FRED_API_KEY set"}), 400
    df = fetch_historical_data(key, years=10)
    if df.empty:
        return jsonify({"error": "No historical data"}), 500
    analytics = MacroAnalytics()
    corr_cols = ["cpi", "core_cpi", "gdp", "nfp", "unemployment",
                  "dxy", "treasury_10y", "consumer_sentiment",
                  "industrial_production", "housing_starts"]
    corr = analytics.correlation_matrix(df, corr_cols)
    if corr.empty:
        return jsonify([])
    return jsonify({
        "columns": list(corr.columns),
        "values": corr.values.tolist(),
    })


# ── Regime History ───────────────────────────────────────────────────

@app.route("/api/regime/history")
def api_regime_history():
    df = MacroAnalytics.load_history()
    if df.empty:
        return jsonify([])
    regimes = MacroAnalytics.regime_breakdown(df)
    if regimes.empty:
        return jsonify([])
    return jsonify(regimes.to_dict(orient="records"))


@app.route("/api/regime/transitions")
def api_regime_transitions():
    df = MacroAnalytics.load_history()
    if df.empty:
        return jsonify({})
    matrix = MacroAnalytics.regime_transition_matrix(df)
    if matrix.empty:
        return jsonify({})
    return jsonify({
        "matrix": matrix.to_dict(),
        "description": "Row = from regime, Col = to regime. Values are transition probabilities (%)"
    })


# ── Anomalies ────────────────────────────────────────────────────────

@app.route("/api/anomalies")
def api_anomalies():
    df = MacroAnalytics.load_history()
    if df.empty:
        return jsonify([])
    anomalies = MacroAnalytics.detect_anomalies(df)
    if anomalies.empty:
        return jsonify([])
    return jsonify(anomalies.tail(50).to_dict(orient="records"))


@app.route("/api/health-score")
def api_health_score():
    df = MacroAnalytics.load_history()
    if df.empty:
        return jsonify({})
    score = MacroAnalytics.composite_health_score(df)
    latest = float(score.iloc[-1]) if not score.empty else 0
    return jsonify({
        "latest_score": round(latest, 1),
        "score_history": score.tail(50).tolist(),
        "interpretation": (
            "> 3: Strong economy, < -3: Weak economy"
        )
    })


# ── Backtest ─────────────────────────────────────────────────────────

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    key = _get_key()
    if not key:
        return jsonify({"error": "No FRED_API_KEY set"}), 400
    body = request.get_json(silent=True) or {}
    years = body.get("years", 15)
    try:
        result = MacroAnalytics.run_backtest(key, years=years)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/detailed", methods=["POST"])
def api_backtest_detailed():
    key = _get_key()
    if not key:
        return jsonify({"error": "No FRED_API_KEY set"}), 400
    body = request.get_json(silent=True) or {}
    years = body.get("years", 15)

    try:
        data_points, prices = prepare_backtest_data(key, years)
        engine = BacktestEngine(data_points, prices)

        strategy = engine.run_all_models()
        benchmark = engine.run_buy_and_hold()
        walkforward = engine.run_walk_forward(120, 60)
        risk_controlled = engine.run_with_risk_controls(8.0, 25.0)
        sized = engine.run_with_position_sizing()

        return jsonify({
            "strategy": {
                "total_return_pct": round(strategy.total_return_pct, 2),
                "annualized_return_pct": round(strategy.annualized_return_pct, 2),
                "volatility_pct": round(strategy.volatility_pct, 2),
                "sharpe_ratio": round(strategy.sharpe_ratio, 3),
                "sortino_ratio": round(strategy.sortino_ratio, 3),
                "calmar_ratio": round(strategy.calmar_ratio, 3),
                "max_drawdown_pct": round(strategy.max_drawdown_pct, 2),
                "var_95_pct": round(strategy.var_95_pct, 2),
                "cvar_95_pct": round(strategy.cvar_95_pct, 2),
                "ulcer_index": round(strategy.ulcer_index, 2),
                "win_rate_pct": round(strategy.win_rate_pct, 1),
                "profit_factor": round(strategy.profit_factor, 2),
                "total_trades": strategy.total_trades,
                "final_value": round(strategy.final_value, 2),
                "equity_curve": [round(v, 2) for v in strategy.equity_curve],
                "drawdown_curve": [round(v, 2) for v in strategy.drawdown_curve],
                "trades": [
                    {"entry": round(t.entry_price, 2), "exit": round(t.exit_price, 2),
                     "signal": t.signal_at_entry, "return_pct": round(t.return_pct, 2)}
                    for t in strategy.trades
                ],
            },
            "benchmark": {
                "total_return_pct": round(benchmark.total_return_pct, 2),
                "sharpe_ratio": round(benchmark.sharpe_ratio, 3),
                "max_drawdown_pct": round(benchmark.max_drawdown_pct, 2),
                "final_value": round(benchmark.final_value, 2),
            },
            "walk_forward": {
                "total_return_pct": round(walkforward.total_return_pct, 2),
                "sharpe_ratio": round(walkforward.sharpe_ratio, 3),
                "max_drawdown_pct": round(walkforward.max_drawdown_pct, 2),
            },
            "risk_controlled": {
                "total_return_pct": round(risk_controlled.total_return_pct, 2),
                "sharpe_ratio": round(risk_controlled.sharpe_ratio, 3),
                "max_drawdown_pct": round(risk_controlled.max_drawdown_pct, 2),
                "total_trades": risk_controlled.total_trades,
            },
            "position_sized": {
                "total_return_pct": round(sized.total_return_pct, 2),
                "sharpe_ratio": round(sized.sharpe_ratio, 3),
                "max_drawdown_pct": round(sized.max_drawdown_pct, 2),
            },
            "outperformance": round(
                strategy.total_return_pct - benchmark.total_return_pct, 2
            ),
            "parameters": {
                "years": years,
                "stop_loss_pct": 8.0,
                "take_profit_pct": 25.0,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Scenario Analysis ───────────────────────────────────────────────

@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    body = request.get_json(silent=True) or {}
    try:
        analyzer = GDPAnalyzer(
            cpi=body.get("cpi", 0),
            core_gdp=body.get("core_gdp", 0),
            nfp=body.get("nfp", 0),
            dxy=body.get("dxy", 0),
            treasury_3m_pct_change=body.get("treasury_3m_change", 0),
            treasury_yield=body.get("treasury_yield", 0),
            unemployment_rate=body.get("unemployment", 0),
            consumer_sentiment=body.get("consumer_sentiment", 0),
            yield_spread_10y2y=body.get("yield_spread_10y2y", 0),
        )
        votes = analyzer.get_model_votes()
        names = ["InflationVeto", "TaylorRule", "DollarStrength",
                 "BondSignal", "CreditSpread", "LaborMarket", "GlobalRisk"]
        return jsonify({
            "signal": analyzer.signal,
            "confidence": analyzer.confidence,
            "ensemble_value": round(analyzer.get_ensemble_numeric_value(), 4),
            "signal_strength": round(analyzer.get_signal_strength(), 4),
            "model_agreement": round(analyzer.get_model_agreement(), 4),
            "dominant_model": analyzer.get_dominant_model(),
            "regime": str(analyzer.get_regime().regime).split(".")[1],
            "reason": analyzer.reason,
            "models": [
                {"name": names[i] if i < len(names) else f"Model {i}",
                 "signal": v.signal, "confidence": v.confidence,
                 "value": round(v.numeric_value, 3), "reason": v.reason}
                for i, v in enumerate(votes)
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Compare Scenarios ───────────────────────────────────────────────

@app.route("/api/compare", methods=["POST"])
def api_compare():
    body = request.get_json(silent=True) or {}
    scenarios = body.get("scenarios", [])
    if not scenarios or not isinstance(scenarios, list):
        return jsonify({"error": "Provide a 'scenarios' array"}), 400

    results = []
    for i, s in enumerate(scenarios):
        try:
            analyzer = GDPAnalyzer(
                cpi=s.get("cpi", 0),
                core_gdp=s.get("core_gdp", 0),
                nfp=s.get("nfp", 0),
                dxy=s.get("dxy", 0),
                treasury_3m_pct_change=s.get("treasury_3m_change", 0),
                treasury_yield=s.get("treasury_yield", 0),
                unemployment_rate=s.get("unemployment", 0),
                consumer_sentiment=s.get("consumer_sentiment", 0),
                yield_spread_10y2y=s.get("yield_spread_10y2y", 0),
            )
            results.append({
                "scenario": s.get("name", f"Scenario {i + 1}"),
                "signal": analyzer.signal,
                "confidence": analyzer.confidence,
                "ensemble_value": round(analyzer.get_ensemble_numeric_value(), 4),
                "regime": str(analyzer.get_regime().regime).split(".")[1],
            })
        except Exception as e:
            results.append({
                "scenario": s.get("name", f"Scenario {i + 1}"),
                "error": str(e),
            })

    return jsonify({"scenarios": results})


# ── Indicators ──────────────────────────────────────────────────────

@app.route("/api/indicators")
def api_indicators():
    return jsonify({
        name: sid for name, sid in sorted(FRED_SERIES.items())
    })


# ── Sensitivity ─────────────────────────────────────────────────────

@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    body = request.get_json(silent=True) or {}
    variable = body.get("variable", "cpi")
    steps = body.get("steps", 11)

    base_data = {
        "cpi": body.get("cpi", 0),
        "core_gdp": body.get("core_gdp", 0),
        "nfp": body.get("nfp", 0),
        "dxy": body.get("dxy", 0),
        "treasury_3m_change": body.get("treasury_3m_change", 0),
        "treasury_yield": body.get("treasury_yield", 0),
        "unemployment": body.get("unemployment", 0),
        "consumer_sentiment": body.get("consumer_sentiment", 0),
        "yield_spread_10y2y": body.get("yield_spread_10y2y", 0),
    }

    result = MacroAnalytics.sensitivity_analysis(base_data, variable, steps=steps)
    if result.empty:
        return jsonify({"error": f"Cannot run sensitivity on '{variable}'"}), 400
    return jsonify(result.to_dict(orient="records"))


# ── Report ──────────────────────────────────────────────────────────

@app.route("/api/report", methods=["POST"])
def api_report():
    key = _get_key()
    if not key:
        return jsonify({"error": "No FRED_API_KEY set"}), 400

    body = request.get_json(silent=True) or {}
    include_backtest = body.get("backtest", False)
    backtest_years = body.get("years", 15)

    data = run_analysis(key)
    analytics = MacroAnalytics()
    report = analytics.full_report(data)

    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "report": report,
    }

    if include_backtest:
        from python.reporting import ResearchReporter
        backtest = MacroAnalytics.run_backtest(key, years=backtest_years)
        history = MacroAnalytics.load_history()
        signal_stats = analytics.signal_accuracy(history) if not history.empty else {}
        result["backtest"] = backtest
        result["signal_statistics"] = signal_stats

        md = ResearchReporter.generate_markdown_report(report, backtest)
        result["markdown"] = md

    return jsonify(result)


if __name__ == "__main__":
    if not os.environ.get("FRED_API_KEY"):
        print("[ERROR] Set FRED_API_KEY in .env file.")
        sys.exit(1)
    from waitress import serve
    print("  GDP Research Engine API v2.0 — http://127.0.0.1:5000")
    print("  7 signal models, 18 endpoints, VaR/CVaR risk metrics")
    print("  Production server (Waitress)")
    serve(app, host="127.0.0.1", port=5000)