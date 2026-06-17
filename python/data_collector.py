import os
import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fredapi import Fred

from gdp_engine import GDPAnalyzer, BacktestEngine, DataPoint

FRED_SERIES = {
    "gdp": "GDPC1",
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "nfp": "PAYEMS",
    "unemployment": "UNRATE",
    "dxy": "DTWEXBGS",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "treasury_3m": "DGS3MO",
    "consumer_sentiment": "UMCSENT",
    "industrial_production": "INDPRO",
    "retail_sales": "RSXFS",
    "housing_starts": "HOUST",
    "pmi": "CUMFNS",  # Capacity Utilization: Manufacturing (PMI proxy)
    "fed_funds": "FEDFUNDS",
    "gdp_deflator": "GDPDEF",
    "personal_income": "PI",
    "personal_savings": "PSAVERT",
    "velocity_m2": "M2V",
    "corp_profits": "CP",
}

PENDING_WINDOW_DAYS = 7
CACHE_DIR = Path("data/cache")


class MacroDataCollector:
    def __init__(self, api_key: str):
        self._fred = Fred(api_key=api_key)

    def _fetch_series(self, series_id: str, cache_name: str = None) -> pd.Series:
        cache_name = cache_name or series_id
        cache_path = CACHE_DIR / f"{cache_name}.parquet"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            last_idx = cached.index[-1]
            if hasattr(last_idx, 'tz') and last_idx.tz is not None:
                now = datetime.now(timezone.utc)
            else:
                now = datetime.utcnow()
            cache_age = now - last_idx
            if cache_age < timedelta(hours=24):
                return cached.iloc[:, 0]

        raw = self._fred.get_series(series_id).dropna().sort_index()
        raw.to_frame("value").to_parquet(cache_path)
        return raw

    def get_gdp_release_date(self):
        try:
            series = self._fred.get_series(FRED_SERIES["gdp"]).dropna()
            if series.empty:
                return None
            return series.index[-1].date()
        except Exception as e:
            print(f"[WARN] Failed to get GDP release date: {e}")
            return None

    def fetch_gdp(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["gdp"], "gdp")
            if len(series) < 2:
                return 0.0
            latest = series.iloc[-1]
            prev = series.iloc[-2]
            quarterly_growth = (latest / prev) - 1.0
            annualized = ((1.0 + quarterly_growth) ** 4 - 1.0) * 100.0
            return round(annualized, 2)
        except Exception as e:
            print(f"[WARN] GDP fetch failed: {e}")
            return 0.0

    def fetch_cpi(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["cpi"], "cpi")
            if len(series) < 13:
                return 0.0
            latest = series.iloc[-1]
            prev_year = series.iloc[-13]
            cpi_yoy = ((latest / prev_year) - 1.0) * 100.0
            return round(cpi_yoy, 2)
        except Exception as e:
            print(f"[WARN] CPI fetch failed: {e}")
            return 0.0

    def fetch_core_cpi(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["core_cpi"], "core_cpi")
            if len(series) < 13:
                return 0.0
            latest = series.iloc[-1]
            prev_year = series.iloc[-13]
            yoy = ((latest / prev_year) - 1.0) * 100.0
            return round(yoy, 2)
        except Exception as e:
            print(f"[WARN] Core CPI fetch failed: {e}")
            return 0.0

    def fetch_nfp(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["nfp"], "nfp")
            if len(series) < 2:
                return 0.0
            latest = series.iloc[-1]
            prev = series.iloc[-2]
            change_thousands = latest - prev
            return round(change_thousands * 1000, 0)
        except Exception as e:
            print(f"[WARN] NFP fetch failed: {e}")
            return 0.0

    def fetch_unemployment(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["unemployment"], "unemployment")
            if series.empty:
                return 0.0
            return round(float(series.iloc[-1]), 1)
        except Exception as e:
            print(f"[WARN] Unemployment fetch failed: {e}")
            return 0.0

    def fetch_dxy(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["dxy"], "dxy")
            if series.empty:
                return 0.0
            return round(float(series.iloc[-1]), 2)
        except Exception as e:
            print(f"[WARN] DXY fetch failed: {e}")
            return 0.0

    def fetch_treasury(self) -> tuple[float, float, float, float]:
        try:
            series_10y = self._fetch_series(FRED_SERIES["treasury_10y"], "treasury_10y")
            series_2y = self._fetch_series(FRED_SERIES["treasury_2y"], "treasury_2y")
            series_3m = self._fetch_series(FRED_SERIES["treasury_3m"], "treasury_3m")

            if len(series_10y) < 2:
                return (0.0, 0.0, 0.0, 0.0)

            ty_current = float(series_10y.iloc[-1])
            t2_current = float(series_2y.iloc[-1]) if len(series_2y) > 0 else 0.0
            lookback = min(63, len(series_10y) - 1)
            ty_old = float(series_10y.iloc[-1 - lookback])
            ty_3m_chg = ((ty_current / ty_old) - 1.0) * 100.0 if ty_old != 0 else 0.0

            spread = ty_current - t2_current if t2_current else 0.0

            return (round(ty_current, 2), round(ty_3m_chg, 2),
                    round(t2_current, 2), round(spread, 2))
        except Exception as e:
            print(f"[WARN] Treasury fetch failed: {e}")
            return (0.0, 0.0, 0.0, 0.0)

    def fetch_consumer_sentiment(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["consumer_sentiment"], "consumer_sentiment")
            if series.empty:
                return 0.0
            return round(float(series.iloc[-1]), 1)
        except Exception as e:
            print(f"[WARN] Consumer sentiment fetch failed: {e}")
            return 0.0

    def fetch_pmi(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["pmi"], "pmi")
            if series.empty:
                return 0.0
            return round(float(series.iloc[-1]), 1)
        except Exception as e:
            print(f"[WARN] PMI fetch failed: {e}")
            return 0.0

    def fetch_fed_funds(self) -> float:
        try:
            series = self._fetch_series(FRED_SERIES["fed_funds"], "fed_funds")
            if series.empty:
                return 0.0
            return round(float(series.iloc[-1]), 2)
        except Exception as e:
            print(f"[WARN] Fed funds fetch failed: {e}")
            return 0.0

    def get_all(self) -> dict:
        dxy = self.fetch_dxy()
        ty, ty_3m, t2y, spread = self.fetch_treasury()
        unemployment = self.fetch_unemployment()
        sentiment = self.fetch_consumer_sentiment()
        pmi = self.fetch_pmi()
        fed_funds = self.fetch_fed_funds()

        return {
            "cpi": self.fetch_cpi(),
            "core_cpi": self.fetch_core_cpi(),
            "core_gdp": self.fetch_gdp(),
            "nfp": self.fetch_nfp(),
            "unemployment": unemployment,
            "dxy": dxy,
            "treasury_yield": ty,
            "treasury_2y": t2y,
            "treasury_3m_change": ty_3m,
            "yield_spread_10y2y": spread,
            "consumer_sentiment": sentiment,
            "pmi": pmi,
            "fed_funds": fed_funds,
        }


def _days_since_gdp(collector: MacroDataCollector) -> tuple[int | None, date | None]:
    release = collector.get_gdp_release_date()
    if release is None:
        return None, None
    today = datetime.utcnow().date()
    return (today - release).days, release


def _apply_pending_rule(
    collector: MacroDataCollector, data: dict,
) -> tuple[str, int, str, int | None]:
    days_since, release = _days_since_gdp(collector)
    if days_since is not None and days_since <= PENDING_WINDOW_DAYS:
        remaining = PENDING_WINDOW_DAYS - days_since
        reason = (
            f"Waiting for NFP confirmation ({remaining} day(s) remaining). "
            f"GDP released {release}. Analysis will finalize after "
            f"{PENDING_WINDOW_DAYS}-day confirmation window."
        )
        return "PENDING", 0, reason, remaining
    signal, confidence, reason = _run_ensemble_analysis(data)
    return signal, confidence, reason, None


def _run_ensemble_analysis(data: dict) -> tuple[str, int, str]:
    analyzer = GDPAnalyzer(
        cpi=data["cpi"],
        core_gdp=data["core_gdp"],
        nfp=data["nfp"],
        dxy=data["dxy"],
        treasury_3m_pct_change=data["treasury_3m_change"],
        treasury_yield=data["treasury_yield"],
        unemployment_rate=data.get("unemployment", 0.0),
        consumer_sentiment=data.get("consumer_sentiment", 0.0),
        yield_spread_10y2y=data.get("yield_spread_10y2y", 0.0),
    )
    return analyzer.signal, analyzer.confidence, analyzer.reason


def run_analysis(api_key: str | None = None) -> dict:
    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("[ERROR] No FRED API key provided.")
        return {}

    print(f"  GDP RESEARCH ENGINE - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    collector = MacroDataCollector(api_key)
    data = collector.get_all()
    signal, confidence, reason, days_remaining = _apply_pending_rule(collector, data)

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        **data,
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "days_remaining": days_remaining,
    }

    _log_to_csv(result)
    _print_dashboard(result)

    return result


def _log_to_csv(result: dict):
    log_path = Path("data") / "analysis_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    log_fields = [
        "timestamp", "cpi", "core_cpi", "core_gdp", "nfp", "unemployment",
        "dxy", "treasury_yield", "treasury_2y", "treasury_3m_change",
        "yield_spread_10y2y", "consumer_sentiment", "pmi", "fed_funds",
        "signal", "confidence", "days_remaining",
    ]

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in log_fields})

    print(f"  Logged to: {log_path}")


def _print_dashboard(result: dict):
    C = {"R": "\033[91m", "G": "\033[92m", "Y": "\033[93m",
         "B": "\033[94m", "M": "\033[95m", "0": "\033[0m"}

    signal = result.get("signal", "—")
    confidence = result.get("confidence", 0)
    days_remaining = result.get("days_remaining")

    signal_color = {"BUY": C["G"], "STRONG_BUY": C["G"],
                    "SELL": C["R"], "STRONG_SELL": C["R"],
                    "HOLD": C["Y"], "PENDING": C["B"]}.get(signal, C["0"])

    ts_display = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    g = result.get("core_gdp", 0) or 0
    c = result.get("cpi", 0) or 0
    n = result.get("nfp", 0) or 0
    d = result.get("dxy", 0) or 0
    ty = result.get("treasury_yield", 0) or 0
    t3 = result.get("treasury_3m_change", 0) or 0
    unemp = result.get("unemployment", 0) or 0
    sent = result.get("consumer_sentiment", 0) or 0
    pmi_val = result.get("pmi", 0) or 0
    spread = result.get("yield_spread_10y2y", 0) or 0
    pmi_str = f"{pmi_val:>4.1f}" if pmi_val else "  N/A"

    gdp_c = C["R"] if g > 2.5 else C["Y"] if g >= 1.5 else C["G"]
    cpi_c = C["R"] if c > 3.5 else C["G"]
    dxy_c = C["R"] if d > 105 else C["0"]
    nfp_c = C["G"] if n > 200_000 else C["R"]

    L = "+" + "-" * 64 + "+"
    print(f"  {L}")
    print(f"  |  GDP RESEARCH ENGINE v2.0 - ENSEMBLE SIGNAL{'':>12}|")
    print(f"  |{'-' * 64}|")
    print(f"  |  {ts_display:<62}|")
    print(f"  |{'-' * 64}|")
    print(f"  |  {gdp_c}GDP     {g:>6.2f}%   CPI  {cpi_c}{c:>5.2f}%   NFP   {nfp_c}{n:>7,.0f}{C['0']}{'':>14}|")
    print(f"  |  DXY    {dxy_c}{d:>6.2f}   Unemp {unemp:>4.1f}%   PMI   {pmi_str}{'':>18}|")
    print(f"  |  10Y    {ty:>5.2f}%   Spread {spread:>+5.2f}%   Sent  {sent:>5.1f}{'':>18}|")
    print(f"  |{'-' * 64}|")
    print(f"  |  Signal      {signal_color}{signal:<20}{C['0']}{'':>28}|")

    if days_remaining is None:
        conf_c = C["G"] if confidence >= 80 else C["Y"] if confidence >= 50 else C["R"]
        print(f"  |  Confidence  {conf_c}{confidence:>3d}%{'':>50}{C['0']}|")
    else:
        print(f"  |  Confidence  --- (pending, {days_remaining}d remain){'':>27}|")
    print(f"  |{'-' * 64}|")

    rl = _wrap(result.get("reason", ""), 58)
    print(f"  |  Reason {rl[0]:<56}|")
    for ln in rl[1:]:
        print(f"  |         {ln:<56}|")
    print(f"  |{'-' * 64}|")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines if lines else [""]


def fetch_historical_data(api_key: str, years: int = 20) -> pd.DataFrame:
    collector = MacroDataCollector(api_key)
    all_data = {}

    for name, series_id in FRED_SERIES.items():
        try:
            raw = collector._fetch_series(series_id, name)
            all_data[name] = raw
        except Exception as e:
            print(f"[WARN] Could not fetch {name}: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df.index >= cutoff]
    return df


def prepare_backtest_data(api_key: str, years: int = 20) -> tuple[list, list]:
    df = fetch_historical_data(api_key, years)
    df = df.dropna(subset=["gdp", "cpi", "nfp"]).copy()
    if df.empty:
        return [], []

    data_points = []
    for _, row in df.iterrows():
        dp = DataPoint(
            gdp=row.get("gdp", 0.0) if pd.notna(row.get("gdp")) else 0.0,
            cpi=row.get("cpi", 0.0) if pd.notna(row.get("cpi")) else 0.0,
            nfp=row.get("nfp", 0.0) if pd.notna(row.get("nfp")) else 0.0,
            dxy=row.get("dxy", 0.0) if pd.notna(row.get("dxy")) else 0.0,
            treasury_yield=row.get("treasury_10y", 0.0) if pd.notna(row.get("treasury_10y")) else 0.0,
            treasury_3m_change=row.get("treasury_3m_change", 0.0) if pd.notna(row.get("treasury_3m_change")) else 0.0,
            unemployment_rate=row.get("unemployment", 0.0) if pd.notna(row.get("unemployment")) else 0.0,
            consumer_sentiment=row.get("consumer_sentiment", 0.0) if pd.notna(row.get("consumer_sentiment")) else 0.0,
            yield_spread_10y2y=row.get("yield_spread_10y2y", 0.0) if pd.notna(row.get("yield_spread_10y2y")) else 0.0,
        )
        data_points.append(dp)

    if "industrial_production" in df.columns and df["industrial_production"].notna().sum() > 10:
        proxy_price = df["industrial_production"].ffill().values.tolist()
    elif "cpi" in df.columns and df["cpi"].notna().sum() > 10:
        proxy_price = df["cpi"].cumsum().ffill().values.tolist()
    else:
        proxy_price = list(range(len(data_points)))

    return data_points, proxy_price


if __name__ == "__main__":
    run_analysis()
