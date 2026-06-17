#pragma once

#include <string>
#include <vector>

namespace gdp_engine {

int add(int a, int b);

// ── Macro Regime Detection ──────────────────────────────────────────

enum class MacroRegime {
    EXPANSION,
    INFLATIONARY,
    RECESSION,
    STAGFLATION,
    NORMAL,
    UNKNOWN
};

std::string regimeToString(MacroRegime r);

struct RegimeResult {
    MacroRegime regime;
    double expansion_prob;
    double inflation_prob;
    double recession_prob;
    double stagflation_prob;
    double normal_prob;
};

class RegimeDetector {
public:
    RegimeDetector(double cpi, double core_gdp, double nfp,
                   double dxy, double treasury_yield,
                   double unemployment_rate, double consumer_sentiment,
                   double yield_spread_10y2y = 0.0);

    [[nodiscard]] RegimeResult analyze() const;

private:
    double m_cpi;
    double m_core_gdp;
    double m_nfp;
    double m_dxy;
    double m_treasury_yield;
    double m_unemployment_rate;
    double m_consumer_sentiment;
    double m_yield_spread;
};

// ── Signal Models ───────────────────────────────────────────────────

struct SignalVote {
    std::string signal;
    int confidence;
    double numeric_value;
    std::string reason;
};

class TaylorRuleModel {
public:
    TaylorRuleModel(double cpi, double core_gdp, double treasury_yield,
                    double unemployment_rate);
    [[nodiscard]] SignalVote evaluate() const;
private:
    double m_cpi, m_core_gdp, m_treasury_yield, m_unemployment_rate;
};

class InflationVetoModel {
public:
    InflationVetoModel(double cpi, double core_gdp, double nfp);
    [[nodiscard]] SignalVote evaluate() const;
private:
    double m_cpi, m_core_gdp, m_nfp;
};

class DollarStrengthModel {
public:
    DollarStrengthModel(double dxy, double treasury_3m_pct_change, double nfp);
    [[nodiscard]] SignalVote evaluate() const;
private:
    double m_dxy, m_treasury_3m, m_nfp;
};

class BondSignalModel {
public:
    BondSignalModel(double treasury_yield, double treasury_3m_pct_change,
                    double cpi, double core_gdp);
    [[nodiscard]] SignalVote evaluate() const;
private:
    double m_treasury_yield, m_treasury_3m, m_cpi, m_core_gdp;
};

// ── New Signal Models ────────────────────────────────────────────────

class CreditSpreadModel {
public:
    CreditSpreadModel(double treasury_yield, double treasury_3m_pct_change,
                      double unemployment_rate, double cpi, double core_gdp);
    [[nodiscard]] SignalVote evaluate() const;
private:
    double m_treasury_yield, m_treasury_3m, m_unemployment_rate, m_cpi, m_core_gdp;
};

class LaborMarketModel {
public:
    LaborMarketModel(double nfp, double unemployment_rate,
                     double consumer_sentiment, double core_gdp);
    [[nodiscard]] SignalVote evaluate() const;
private:
    double m_nfp, m_unemployment_rate, m_consumer_sentiment, m_core_gdp;
};

class GlobalRiskModel {
public:
    GlobalRiskModel(double dxy, double consumer_sentiment,
                    double treasury_3m_pct_change, double cpi);
    [[nodiscard]] SignalVote evaluate() const;
private:
    double m_dxy, m_consumer_sentiment, m_treasury_3m, m_cpi;
};

// ── GDP Analyzer ────────────────────────────────────────────────────

class GDPAnalyzer {
public:
    GDPAnalyzer(double cpi, double core_gdp, double nfp,
                double dxy, double treasury_3m_pct_change,
                double treasury_yield = 0.0,
                double unemployment_rate = 0.0,
                double consumer_sentiment = 0.0,
                double yield_spread_10y2y = 0.0);

    [[nodiscard]] std::string getSignal() const;
    [[nodiscard]] int getConfidence() const;
    [[nodiscard]] std::string getReason() const;
    [[nodiscard]] RegimeResult getRegime() const;
    [[nodiscard]] std::vector<SignalVote> getModelVotes() const;
    [[nodiscard]] double getEnsembleNumericValue() const;
    [[nodiscard]] std::string getMethodologySummary() const;

    // New research-grade methods
    [[nodiscard]] double getSignalStrength() const;
    [[nodiscard]] std::string getDominantModelName() const;
    [[nodiscard]] double getModelAgreement() const;

private:
    double m_cpi, m_core_gdp, m_nfp, m_dxy;
    double m_treasury_3m, m_treasury_yield;
    double m_unemployment_rate, m_consumer_sentiment, m_yield_spread;
};

// ── Backtest Types ───────────────────────────────────────────────────

struct DataPoint {
    double gdp;
    double cpi;
    double nfp;
    double dxy;
    double treasury_yield;
    double treasury_3m_change;
    double unemployment_rate;
    double consumer_sentiment;
    double yield_spread_10y2y;
};

struct TradeRecord {
    double entry_price;
    double exit_price;
    std::string signal_at_entry;
    double return_pct;

    [[nodiscard]] bool isWin() const noexcept {
        return return_pct > 0.0;
    }
};

struct VaRResult {
    double var_95;
    double var_99;
    double cvar_95;
    double cvar_99;
    double historical_var_95;
    double historical_var_99;
};

struct EnhancedBacktestResult {
    double total_return_pct;
    double annualized_return_pct;
    double cumulative_return_pct;
    double volatility_pct;
    double max_drawdown_pct;
    double downside_deviation_pct;
    double sharpe_ratio;
    double sortino_ratio;
    double calmar_ratio;
    double information_ratio;
    double treynor_ratio;
    double beta;
    double alpha;
    double var_95_pct;
    double cvar_95_pct;
    double ulcer_index;

    int total_trades;
    int winning_trades;
    int losing_trades;
    double win_rate_pct;
    double profit_factor;
    double avg_win_pct;
    double avg_loss_pct;
    double largest_win_pct;
    double largest_loss_pct;
    double avg_hold_bars;

    double initial_capital;
    double final_value;
    double total_fees;

    std::vector<double> equity_curve;
    std::vector<double> drawdown_curve;
    std::vector<TradeRecord> trades;
};

// ── Backtest Engine ──────────────────────────────────────────────────

class BacktestEngine {
public:
    BacktestEngine(std::vector<DataPoint> data, std::vector<double> prices);

    [[nodiscard]] EnhancedBacktestResult runAllModels() const;
    [[nodiscard]] EnhancedBacktestResult runWithModel(const std::string& model_name) const;
    [[nodiscard]] EnhancedBacktestResult runWalkForward(size_t train_size, size_t test_size) const;
    [[nodiscard]] EnhancedBacktestResult runBuyAndHold() const;

    // New enhanced methods
    [[nodiscard]] EnhancedBacktestResult runWithRiskControls(
        double stop_loss_pct, double take_profit_pct) const;
    [[nodiscard]] EnhancedBacktestResult runWithPositionSizing() const;

private:
    [[nodiscard]] static std::string signalFor(double cpi, double gdp, double nfp,
                                                double dxy, double treasury_3m,
                                                double unemployment_rate,
                                                double consumer_sentiment,
                                                double yield_spread) noexcept;
    [[nodiscard]] static EnhancedBacktestResult computeMetrics(
        const std::vector<double>& equity, const std::vector<TradeRecord>& trades,
        double initial_capital) noexcept;

    std::vector<DataPoint> m_data;
    std::vector<double> m_prices;
};

// ── Statistical Utilities ───────────────────────────────────────────

struct RollingStats {
    std::vector<double> rolling_mean;
    std::vector<double> rolling_std;
    std::vector<double> rolling_zscore;
    std::vector<double> rolling_sharpe;
};

class StatisticsEngine {
public:
    static double mean(const std::vector<double>& values) noexcept;
    static double stddev(const std::vector<double>& values) noexcept;
    static double correlation(const std::vector<double>& a,
                              const std::vector<double>& b) noexcept;
    static double sharpeRatio(const std::vector<double>& returns,
                              double risk_free_rate = 0.05) noexcept;
    static double sortinoRatio(const std::vector<double>& returns,
                               double risk_free_rate = 0.05) noexcept;
    static double maxDrawdown(const std::vector<double>& equity) noexcept;
    static double calmarRatio(const std::vector<double>& returns,
                              const std::vector<double>& equity) noexcept;
    static RollingStats rollingStats(const std::vector<double>& values,
                                     size_t window) noexcept;

    // New risk metrics
    static double valueAtRisk(const std::vector<double>& returns,
                              double confidence = 0.95) noexcept;
    static double conditionalVaR(const std::vector<double>& returns,
                                 double confidence = 0.95) noexcept;
    static double historicalVaR(const std::vector<double>& returns,
                                double confidence = 0.95) noexcept;
    static double informationRatio(const std::vector<double>& returns,
                                   const std::vector<double>& benchmark) noexcept;
    static double treynorRatio(const std::vector<double>& returns,
                               const std::vector<double>& benchmark,
                               double risk_free_rate = 0.05) noexcept;
    static double beta(const std::vector<double>& returns,
                       const std::vector<double>& benchmark) noexcept;
    static double alpha(const std::vector<double>& returns,
                        const std::vector<double>& benchmark,
                        double risk_free_rate = 0.05) noexcept;
    static double ulcerIndex(const std::vector<double>& equity) noexcept;
    static double gainLossRatio(const std::vector<double>& returns) noexcept;
};

} // namespace gdp_engine