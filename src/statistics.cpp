#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "gdp_engine.hpp"

namespace gdp_engine {

double StatisticsEngine::mean(const std::vector<double>& values) noexcept {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double StatisticsEngine::stddev(const std::vector<double>& values) noexcept {
    if (values.size() < 2) return 0.0;
    double m = mean(values);
    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - m) * (v - m);
    }
    return std::sqrt(sq_sum / values.size());
}

double StatisticsEngine::correlation(const std::vector<double>& a,
                                      const std::vector<double>& b) noexcept {
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0;

    double ma = 0.0, mb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;

    double cov = 0.0, va = 0.0, vb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double da = a[i] - ma;
        double db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }

    double denom = std::sqrt(va * vb);
    return denom > 0.0 ? cov / denom : 0.0;
}

double StatisticsEngine::sharpeRatio(const std::vector<double>& returns,
                                      double risk_free_rate) noexcept {
    if (returns.size() < 2) return 0.0;

    double rf_per_period = risk_free_rate / 252.0;
    std::vector<double> excess;
    excess.reserve(returns.size());
    for (double r : returns) {
        excess.push_back(r - rf_per_period);
    }

    double mean_excess = mean(excess);
    double sd = stddev(excess);

    if (sd == 0.0) return 0.0;
    return (mean_excess / sd) * std::sqrt(252.0);
}

double StatisticsEngine::sortinoRatio(const std::vector<double>& returns,
                                       double risk_free_rate) noexcept {
    if (returns.size() < 2) return 0.0;

    double rf_per_period = risk_free_rate / 252.0;
    double mean_excess = mean(returns) - rf_per_period;

    double downside_var = 0.0;
    int downside_count = 0;
    for (double r : returns) {
        double excess = r - rf_per_period;
        if (excess < 0) {
            downside_var += excess * excess;
            ++downside_count;
        }
    }

    if (downside_count == 0) return mean_excess > 0.0 ? 999.9 : 0.0;
    double downside_dev = std::sqrt(downside_var / returns.size());

    return downside_dev > 0.0 ? (mean_excess / downside_dev) * std::sqrt(252.0) : 0.0;
}

double StatisticsEngine::maxDrawdown(const std::vector<double>& equity) noexcept {
    if (equity.empty()) return 0.0;
    double peak = equity[0];
    double max_dd = 0.0;
    for (double v : equity) {
        peak = std::max(peak, v);
        max_dd = std::min(max_dd, (v - peak) / peak);
    }
    return max_dd * 100.0;
}

double StatisticsEngine::calmarRatio(const std::vector<double>& returns,
                                       const std::vector<double>& equity) noexcept {
    if (returns.size() < 2 || equity.empty()) return 0.0;

    double total_ratio = equity.back() / equity.front();
    double years = static_cast<double>(returns.size()) / 252.0;
    double ann_return = (std::pow(total_ratio, 1.0 / std::max(years, 0.01)) - 1.0);

    double max_dd = maxDrawdown(equity) / 100.0;

    return max_dd != 0.0 ? ann_return / std::abs(max_dd) : 0.0;
}

// ── Parametric VaR (assumes normal distribution) ────────────────────

double StatisticsEngine::valueAtRisk(const std::vector<double>& returns,
                                      double confidence) noexcept {
    if (returns.size() < 2) return 0.0;

    double m = mean(returns);
    double sd = stddev(returns);

    // Inverse normal approximation (z-score)
    double z = 0.0;
    if (confidence >= 0.99) z = 2.326;
    else if (confidence >= 0.975) z = 1.96;
    else if (confidence >= 0.95) z = 1.645;
    else if (confidence >= 0.90) z = 1.282;
    else z = 1.0;

    return m - z * sd;
}

// ── Historical VaR ──────────────────────────────────────────────────

double StatisticsEngine::historicalVaR(const std::vector<double>& returns,
                                        double confidence) noexcept {
    if (returns.empty()) return 0.0;

    std::vector<double> sorted = returns;
    std::sort(sorted.begin(), sorted.end());

    size_t idx = static_cast<size_t>((1.0 - confidence) * sorted.size());
    idx = std::min(idx, sorted.size() - 1);

    return sorted[idx];
}

// ── Conditional VaR (Expected Shortfall) ────────────────────────────

double StatisticsEngine::conditionalVaR(const std::vector<double>& returns,
                                         double confidence) noexcept {
    if (returns.size() < 2) return 0.0;

    std::vector<double> sorted = returns;
    std::sort(sorted.begin(), sorted.end());

    size_t tail_idx = static_cast<size_t>((1.0 - confidence) * sorted.size());
    tail_idx = std::min(tail_idx, sorted.size() - 1);
    if (tail_idx < 1) tail_idx = 1;

    double sum = 0.0;
    for (size_t i = 0; i < tail_idx; ++i) {
        sum += sorted[i];
    }

    return sum / static_cast<double>(tail_idx);
}

// ── Information Ratio ───────────────────────────────────────────────

double StatisticsEngine::informationRatio(const std::vector<double>& returns,
                                           const std::vector<double>& benchmark) noexcept {
    size_t n = std::min(returns.size(), benchmark.size());
    if (n < 2) return 0.0;

    std::vector<double> diff(n);
    for (size_t i = 0; i < n; ++i) {
        diff[i] = returns[i] - benchmark[i];
    }

    double mean_diff = mean(diff);
    double sd_diff = stddev(diff);

    return sd_diff > 0.0 ? (mean_diff / sd_diff) * std::sqrt(252.0) : 0.0;
}

// ── Treynor Ratio ───────────────────────────────────────────────────

double StatisticsEngine::treynorRatio(const std::vector<double>& returns,
                                       const std::vector<double>& benchmark,
                                       double risk_free_rate) noexcept {
    double b = beta(returns, benchmark);
    if (b == 0.0) return 0.0;

    double mean_return = mean(returns) * 252.0;
    return (mean_return - risk_free_rate) / b;
}

// ── Beta ────────────────────────────────────────────────────────────

double StatisticsEngine::beta(const std::vector<double>& returns,
                               const std::vector<double>& benchmark) noexcept {
    size_t n = std::min(returns.size(), benchmark.size());
    if (n < 2) return 0.0;

    double mr = mean(returns);
    double mb = mean(benchmark);

    double cov = 0.0, var_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dr = returns[i] - mr;
        double db = benchmark[i] - mb;
        cov += dr * db;
        var_b += db * db;
    }

    return var_b > 0.0 ? cov / var_b : 0.0;
}

// ── Alpha (Jensen's) ────────────────────────────────────────────────

double StatisticsEngine::alpha(const std::vector<double>& returns,
                                const std::vector<double>& benchmark,
                                double risk_free_rate) noexcept {
    double b = beta(returns, benchmark);
    double mean_ret = mean(returns) * 252.0;
    double mean_bench = mean(benchmark) * 252.0;

    return mean_ret - (risk_free_rate + b * (mean_bench - risk_free_rate));
}

// ── Ulcer Index ─────────────────────────────────────────────────────

double StatisticsEngine::ulcerIndex(const std::vector<double>& equity) noexcept {
    if (equity.size() < 2) return 0.0;

    double peak = equity[0];
    double sum_sq = 0.0;
    size_t n = equity.size();

    for (double v : equity) {
        peak = std::max(peak, v);
        double pct_dd = ((v - peak) / peak) * 100.0;
        sum_sq += pct_dd * pct_dd;
    }

    return std::sqrt(sum_sq / n);
}

// ── Gain/Loss Ratio ─────────────────────────────────────────────────

double StatisticsEngine::gainLossRatio(const std::vector<double>& returns) noexcept {
    double gains = 0.0, losses = 0.0;
    int gain_count = 0, loss_count = 0;

    for (double r : returns) {
        if (r > 0) { gains += r; ++gain_count; }
        else if (r < 0) { losses += std::abs(r); ++loss_count; }
    }

    if (loss_count == 0 || loss_count == 0) return 0.0;
    double avg_gain = gains / gain_count;
    double avg_loss = losses / loss_count;

    return avg_loss > 0.0 ? avg_gain / avg_loss : 0.0;
}

RollingStats StatisticsEngine::rollingStats(const std::vector<double>& values,
                                             size_t window) noexcept {
    RollingStats stats;
    size_t n = values.size();
    if (n < window || window == 0) return stats;

    stats.rolling_mean.reserve(n);
    stats.rolling_std.reserve(n);
    stats.rolling_zscore.reserve(n);
    stats.rolling_sharpe.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        size_t start = i >= window ? i - window + 1 : 0;
        size_t count = i - start + 1;

        if (count < 2) {
            stats.rolling_mean.push_back(values[i]);
            stats.rolling_std.push_back(0.0);
            stats.rolling_zscore.push_back(0.0);
            stats.rolling_sharpe.push_back(0.0);
            continue;
        }

        double sum = 0.0;
        for (size_t j = start; j <= i; ++j) sum += values[j];
        double m = sum / count;
        stats.rolling_mean.push_back(m);

        double sq = 0.0;
        for (size_t j = start; j <= i; ++j) sq += (values[j] - m) * (values[j] - m);
        double sd = std::sqrt(sq / count);
        stats.rolling_std.push_back(sd);

        double z = sd > 0.0 ? (values[i] - m) / sd : 0.0;
        stats.rolling_zscore.push_back(z);

        double sharpe = sd > 0.0 ? (m / sd) * std::sqrt(252.0) : 0.0;
        stats.rolling_sharpe.push_back(sharpe);
    }

    return stats;
}

} // namespace gdp_engine