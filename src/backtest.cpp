#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include "gdp_engine.hpp"

namespace gdp_engine {

// ── Enhanced Signal Logic ───────────────────────────────────────────

std::string BacktestEngine::signalFor(double cpi, double gdp, double nfp,
                                       double dxy, double treasury_3m,
                                       double unemployment_rate,
                                       double consumer_sentiment,
                                       double yield_spread) noexcept {
    double score = 0.0;

    // Inflation
    if (cpi > 4.5) score -= 2.0;
    else if (cpi > 3.5) score -= 1.5;
    else if (cpi > 2.5) score -= 0.5;
    else score += 1.0;

    // GDP growth
    if (gdp < 1.0) score += 1.5;
    else if (gdp < 1.5) score += 1.0;
    else if (gdp <= 2.5) score += 0.0;
    else if (gdp <= 3.5) score -= 1.0;
    else score -= 1.5;

    // Labor market
    if (nfp < 100000) score += 1.5;
    else if (nfp < 150000) score += 1.0;
    else if (nfp < 250000) score += 0.0;
    else if (nfp < 300000) score -= 1.0;
    else score -= 1.5;

    // Dollar
    if (dxy > 115.0) score -= 1.5;
    else if (dxy > 105.0) score -= 1.0;
    else if (dxy > 100.0) score -= 0.5;
    else if (dxy < 95.0) score += 1.0;

    // Bond momentum
    if (treasury_3m > 25.0) score -= 1.5;
    else if (treasury_3m > 15.0) score -= 1.0;
    else if (treasury_3m > 5.0) score -= 0.5;

    // Unemployment
    if (unemployment_rate > 6.0) score += 1.5;
    else if (unemployment_rate > 5.0) score += 0.5;
    else if (unemployment_rate < 4.0) score -= 0.5;

    // Consumer sentiment
    if (consumer_sentiment > 85.0) score -= 0.5;
    else if (consumer_sentiment < 60.0) score += 0.5;

    // Yield curve inversion
    if (yield_spread < -0.4) score -= 1.0;

    if (score >= 3.0) return "STRONG_BUY";
    if (score >= 1.0) return "BUY";
    if (score <= -3.0) return "STRONG_SELL";
    if (score <= -1.0) return "SELL";
    return "HOLD";
}

// ── Constructor ─────────────────────────────────────────────────────

BacktestEngine::BacktestEngine(std::vector<DataPoint> data, std::vector<double> prices)
    : m_data(std::move(data)), m_prices(std::move(prices)) {}

// ── Buy & Hold ──────────────────────────────────────────────────────

EnhancedBacktestResult BacktestEngine::runBuyAndHold() const {
    constexpr double INITIAL_CAPITAL = 10000.0;
    std::vector<double> equity;
    std::vector<TradeRecord> trades;

    if (m_prices.empty()) {
        return computeMetrics(equity, trades, INITIAL_CAPITAL);
    }

    double shares = INITIAL_CAPITAL / m_prices.front();
    for (size_t i = 0; i < m_prices.size(); ++i) {
        equity.push_back(shares * m_prices[i]);
    }

    trades.push_back({m_prices.front(), m_prices.back(), "BUY", 0.0});
    trades.back().return_pct = trades.back().isWin() ? 1.0 : -1.0;

    return computeMetrics(equity, trades, INITIAL_CAPITAL);
}

// ── Walk-Forward ────────────────────────────────────────────────────

EnhancedBacktestResult BacktestEngine::runWalkForward(size_t train_size, size_t test_size) const {
    constexpr double INITIAL_CAPITAL = 10000.0;
    std::vector<double> combined_equity = {INITIAL_CAPITAL};
    std::vector<TradeRecord> all_trades;

    size_t pos = 0;
    while (pos + train_size + test_size <= m_data.size()) {
        double cash = combined_equity.back();
        double shares = cash / m_prices[pos + train_size];
        bool in_market = false;
        double entry_price = 0.0;

        for (size_t i = pos + train_size; i < pos + train_size + test_size && i < m_data.size(); ++i) {
            const auto& dp = m_data[i];
            const double price = m_prices[i];
            const std::string sig = signalFor(dp.cpi, dp.gdp, dp.nfp, dp.dxy,
                                              dp.treasury_3m_change,
                                              dp.unemployment_rate,
                                              dp.consumer_sentiment,
                                              dp.yield_spread_10y2y);
            const bool buy = (sig == "BUY" || sig == "STRONG_BUY");
            const bool sell = (sig == "SELL" || sig == "STRONG_SELL");

            if (buy && !in_market) {
                shares = cash / price;
                cash = 0.0;
                entry_price = price;
                in_market = true;
            } else if (sell && in_market) {
                cash = shares * price;
                TradeRecord t{entry_price, price, sig, 0.0};
                t.return_pct = ((price - entry_price) / entry_price) * 100.0;
                all_trades.push_back(t);
                shares = 0.0;
                in_market = false;
            }
            combined_equity.push_back(in_market ? shares * price : cash);
        }

        if (in_market && pos + train_size + test_size <= m_prices.size()) {
            double price = m_prices[std::min(pos + train_size + test_size, m_prices.size()) - 1];
            cash = shares * price;
            TradeRecord t{entry_price, price, "CLOSE", 0.0};
            t.return_pct = ((price - entry_price) / entry_price) * 100.0;
            all_trades.push_back(t);
        }

        pos += test_size;
    }

    return computeMetrics(combined_equity, all_trades, INITIAL_CAPITAL);
}

// ── Run All Models ──────────────────────────────────────────────────

EnhancedBacktestResult BacktestEngine::runAllModels() const {
    constexpr double INITIAL_CAPITAL = 10000.0;
    const size_t n = m_data.size();

    double cash = INITIAL_CAPITAL;
    double shares = 0.0;
    double entry_price = 0.0;

    std::vector<TradeRecord> trades;
    std::vector<double> equity = {INITIAL_CAPITAL};
    bool in_market = false;

    for (size_t i = 0; i < n; ++i) {
        const auto& dp = m_data[i];
        const double price = m_prices[i];
        const std::string sig = signalFor(dp.cpi, dp.gdp, dp.nfp, dp.dxy,
                                          dp.treasury_3m_change,
                                          dp.unemployment_rate,
                                          dp.consumer_sentiment,
                                          dp.yield_spread_10y2y);
        const bool buy = (sig == "BUY" || sig == "STRONG_BUY");
        const bool sell = (sig == "SELL" || sig == "STRONG_SELL");

        if (buy && !in_market) {
            shares = cash / price;
            cash = 0.0;
            entry_price = price;
            in_market = true;
        } else if (sell && in_market) {
            cash = shares * price;
            TradeRecord t{entry_price, price, sig, 0.0};
            t.return_pct = ((price - entry_price) / entry_price) * 100.0;
            trades.push_back(t);
            shares = 0.0;
            in_market = false;
        }

        equity.push_back(in_market ? shares * price : cash);
    }

    if (in_market && !m_prices.empty()) {
        const double price = m_prices.back();
        cash = shares * price;
        TradeRecord t{entry_price, price, "CLOSE", 0.0};
        t.return_pct = ((price - entry_price) / entry_price) * 100.0;
        trades.push_back(t);
        shares = 0.0;
    }

    equity.back() = cash;

    return computeMetrics(equity, trades, INITIAL_CAPITAL);
}

// ── Run with Risk Controls (NEW) ────────────────────────────────────

EnhancedBacktestResult BacktestEngine::runWithRiskControls(
    double stop_loss_pct, double take_profit_pct) const
{
    constexpr double INITIAL_CAPITAL = 10000.0;
    const size_t n = m_data.size();

    double cash = INITIAL_CAPITAL;
    double shares = 0.0;
    double entry_price = 0.0;

    std::vector<TradeRecord> trades;
    std::vector<double> equity = {INITIAL_CAPITAL};
    bool in_market = false;
    double peak_entry = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const auto& dp = m_data[i];
        const double price = m_prices[i];
        const std::string sig = signalFor(dp.cpi, dp.gdp, dp.nfp, dp.dxy,
                                          dp.treasury_3m_change,
                                          dp.unemployment_rate,
                                          dp.consumer_sentiment,
                                          dp.yield_spread_10y2y);
        const bool buy = (sig == "BUY" || sig == "STRONG_BUY");
        const bool sell = (sig == "SELL" || sig == "STRONG_SELL");

        if (in_market) {
            peak_entry = std::max(peak_entry, price);
            double pnl_pct = (price - entry_price) / entry_price * 100.0;
            double drawdown_from_peak = (price - peak_entry) / peak_entry * 100.0;

            if (drawdown_from_peak <= -stop_loss_pct) {
                cash = shares * price;
                TradeRecord t{entry_price, price, "STOP_LOSS", 0.0};
                t.return_pct = pnl_pct;
                trades.push_back(t);
                shares = 0.0;
                in_market = false;
            } else if (take_profit_pct > 0 && pnl_pct >= take_profit_pct) {
                cash = shares * price;
                TradeRecord t{entry_price, price, "TAKE_PROFIT", 0.0};
                t.return_pct = pnl_pct;
                trades.push_back(t);
                shares = 0.0;
                in_market = false;
            }
        }

        if (buy && !in_market) {
            shares = cash / price;
            cash = 0.0;
            entry_price = price;
            peak_entry = price;
            in_market = true;
        } else if (sell && in_market) {
            cash = shares * price;
            double pnl_pct = ((price - entry_price) / entry_price) * 100.0;
            TradeRecord t{entry_price, price, sig, 0.0};
            t.return_pct = pnl_pct;
            trades.push_back(t);
            shares = 0.0;
            in_market = false;
        }

        equity.push_back(in_market ? shares * price : cash);
    }

    if (in_market && !m_prices.empty()) {
        const double price = m_prices.back();
        cash = shares * price;
        double pnl_pct = ((price - entry_price) / entry_price) * 100.0;
        TradeRecord t{entry_price, price, "CLOSE", 0.0};
        t.return_pct = pnl_pct;
        trades.push_back(t);
        shares = 0.0;
    }

    equity.back() = cash;
    return computeMetrics(equity, trades, INITIAL_CAPITAL);
}

// ── Run with Position Sizing (NEW) ──────────────────────────────────

EnhancedBacktestResult BacktestEngine::runWithPositionSizing() const {
    constexpr double INITIAL_CAPITAL = 10000.0;
    const size_t n = m_data.size();

    double cash = INITIAL_CAPITAL;
    double shares = 0.0;
    double entry_price = 0.0;

    std::vector<TradeRecord> trades;
    std::vector<double> equity = {INITIAL_CAPITAL};
    bool in_market = false;

    for (size_t i = 0; i < n; ++i) {
        const auto& dp = m_data[i];
        const double price = m_prices[i];
        const std::string sig = signalFor(dp.cpi, dp.gdp, dp.nfp, dp.dxy,
                                          dp.treasury_3m_change,
                                          dp.unemployment_rate,
                                          dp.consumer_sentiment,
                                          dp.yield_spread_10y2y);

        double confidence = 0.5;
        if (sig == "STRONG_BUY") confidence = 1.0;
        else if (sig == "BUY") confidence = 0.75;
        else if (sig == "SELL") confidence = 0.75;
        else if (sig == "STRONG_SELL") confidence = 1.0;

        const bool buy = (sig == "BUY" || sig == "STRONG_BUY");
        const bool sell = (sig == "SELL" || sig == "STRONG_SELL");

        if (buy && !in_market) {
            double allocated = cash * confidence;
            shares = allocated / price;
            cash -= allocated;
            entry_price = price;
            in_market = true;
        } else if (sell && in_market) {
            cash += shares * price;
            double pnl_pct = ((price - entry_price) / entry_price) * 100.0;
            TradeRecord t{entry_price, price, sig, 0.0};
            t.return_pct = pnl_pct;
            trades.push_back(t);
            shares = 0.0;
            in_market = false;
        }

        equity.push_back(in_market ? shares * price : cash);
    }

    if (in_market && !m_prices.empty()) {
        const double price = m_prices.back();
        cash += shares * price;
        double pnl_pct = ((price - entry_price) / entry_price) * 100.0;
        TradeRecord t{entry_price, price, "CLOSE", 0.0};
        t.return_pct = pnl_pct;
        trades.push_back(t);
        shares = 0.0;
    }

    equity.back() = cash;
    return computeMetrics(equity, trades, INITIAL_CAPITAL);
}

// ── Run with Model ──────────────────────────────────────────────────

EnhancedBacktestResult BacktestEngine::runWithModel(const std::string& model_name) const {
    return runAllModels();
}

// ── Metrics Computation ─────────────────────────────────────────────

EnhancedBacktestResult BacktestEngine::computeMetrics(
    const std::vector<double>& equity, const std::vector<TradeRecord>& trades,
    double initial_capital) noexcept
{
    EnhancedBacktestResult r{};
    r.initial_capital = initial_capital;
    r.trades = trades;

    if (equity.empty()) return r;

    r.equity_curve = equity;
    r.final_value = equity.back();
    r.total_return_pct = ((equity.back() / initial_capital) - 1.0) * 100.0;

    size_t n = equity.size();
    std::vector<double> returns;
    returns.reserve(n - 1);
    for (size_t i = 1; i < n; ++i) {
        returns.push_back((equity[i] - equity[i - 1]) / equity[i - 1]);
    }

    if (returns.empty()) return r;

    double total_ratio = equity.back() / initial_capital;
    double years = static_cast<double>(n) / 252.0;
    r.annualized_return_pct = (std::pow(total_ratio, 1.0 / std::max(years, 0.01)) - 1.0) * 100.0;

    double mean_ret = StatisticsEngine::mean(returns);
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_ret) * (ret - mean_ret);
    }
    variance /= returns.size();
    r.volatility_pct = std::sqrt(variance * 252.0) * 100.0;

    double peak = equity[0];
    double max_dd = 0.0;
    std::vector<double> drawdowns;
    drawdowns.reserve(n);
    for (double val : equity) {
        peak = std::max(peak, val);
        double dd = (val - peak) / peak;
        drawdowns.push_back(dd * 100.0);
        max_dd = std::min(max_dd, dd);
    }
    r.drawdown_curve = drawdowns;
    r.max_drawdown_pct = max_dd * 100.0;

    r.sharpe_ratio = StatisticsEngine::sharpeRatio(returns);
    r.sortino_ratio = StatisticsEngine::sortinoRatio(returns);
    r.calmar_ratio = StatisticsEngine::calmarRatio(returns, equity);
    r.information_ratio = 0.0;
    r.treynor_ratio = 0.0;
    r.beta = 0.0;
    r.alpha = 0.0;

    double downside_var = 0.0;
    for (double ret : returns) {
        if (ret < 0) downside_var += ret * ret;
    }
    downside_var /= returns.size();
    r.downside_deviation_pct = std::sqrt(downside_var * 252.0) * 100.0;

    // VaR and CVaR
    r.var_95_pct = StatisticsEngine::valueAtRisk(returns, 0.95) * 100.0;
    r.cvar_95_pct = StatisticsEngine::conditionalVaR(returns, 0.95) * 100.0;

    // Ulcer index
    r.ulcer_index = StatisticsEngine::ulcerIndex(equity);

    // Trade stats
    r.total_trades = static_cast<int>(trades.size());
    r.winning_trades = 0;
    r.losing_trades = 0;

    double gross_wins = 0.0;
    double gross_losses = 0.0;
    r.largest_win_pct = 0.0;
    r.largest_loss_pct = 0.0;

    for (const auto& t : trades) {
        if (t.isWin()) {
            ++r.winning_trades;
            gross_wins += t.return_pct;
            r.largest_win_pct = std::max(r.largest_win_pct, t.return_pct);
        } else {
            ++r.losing_trades;
            gross_losses += std::abs(t.return_pct);
            r.largest_loss_pct = std::min(r.largest_loss_pct, t.return_pct);
        }
    }

    r.win_rate_pct = r.total_trades > 0
        ? (static_cast<double>(r.winning_trades) / r.total_trades) * 100.0
        : 0.0;

    r.profit_factor = gross_losses > 0.0 ? gross_wins / gross_losses : (gross_wins > 0.0 ? 999.9 : 0.0);

    r.avg_win_pct = r.winning_trades > 0 ? gross_wins / r.winning_trades : 0.0;
    r.avg_loss_pct = r.losing_trades > 0 ? -(gross_losses / r.losing_trades) : 0.0;

    return r;
}

} // namespace gdp_engine