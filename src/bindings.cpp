#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gdp_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gdp_engine, m) {
    m.doc() = "GDP Analysis Engine v2.0 - Research-grade macro signal platform";

    m.def("add", &gdp_engine::add,
          py::arg("a"), py::arg("b"),
          "Add two integers and return the result");

    // ── Macro Regime ─────────────────────────────────────────────
    py::enum_<gdp_engine::MacroRegime>(m, "MacroRegime")
        .value("EXPANSION", gdp_engine::MacroRegime::EXPANSION)
        .value("INFLATIONARY", gdp_engine::MacroRegime::INFLATIONARY)
        .value("RECESSION", gdp_engine::MacroRegime::RECESSION)
        .value("STAGFLATION", gdp_engine::MacroRegime::STAGFLATION)
        .value("NORMAL", gdp_engine::MacroRegime::NORMAL)
        .value("UNKNOWN", gdp_engine::MacroRegime::UNKNOWN)
        .export_values();

    m.def("regime_to_string", &gdp_engine::regimeToString);

    py::class_<gdp_engine::RegimeResult>(m, "RegimeResult")
        .def_readonly("regime", &gdp_engine::RegimeResult::regime)
        .def_readonly("expansion_prob", &gdp_engine::RegimeResult::expansion_prob)
        .def_readonly("inflation_prob", &gdp_engine::RegimeResult::inflation_prob)
        .def_readonly("recession_prob", &gdp_engine::RegimeResult::recession_prob)
        .def_readonly("stagflation_prob", &gdp_engine::RegimeResult::stagflation_prob)
        .def_readonly("normal_prob", &gdp_engine::RegimeResult::normal_prob)
        .def("__repr__", [](const gdp_engine::RegimeResult& r) {
            return "<RegimeResult: " + gdp_engine::regimeToString(r.regime) + ">";
        });

    py::class_<gdp_engine::RegimeDetector>(m, "RegimeDetector")
        .def(py::init<double, double, double, double, double, double, double, double>(),
             py::arg("cpi"), py::arg("core_gdp"), py::arg("nfp"),
             py::arg("dxy"), py::arg("treasury_yield"),
             py::arg("unemployment_rate"), py::arg("consumer_sentiment"),
             py::arg("yield_spread_10y2y") = 0.0)
        .def("analyze", &gdp_engine::RegimeDetector::analyze,
             "Detect the current macro regime with probabilities");

    // ── Signal Models ────────────────────────────────────────────
    py::class_<gdp_engine::SignalVote>(m, "SignalVote")
        .def_readonly("signal", &gdp_engine::SignalVote::signal)
        .def_readonly("confidence", &gdp_engine::SignalVote::confidence)
        .def_readonly("numeric_value", &gdp_engine::SignalVote::numeric_value)
        .def_readonly("reason", &gdp_engine::SignalVote::reason);

    py::class_<gdp_engine::TaylorRuleModel>(m, "TaylorRuleModel")
        .def(py::init<double, double, double, double>(),
             py::arg("cpi"), py::arg("core_gdp"),
             py::arg("treasury_yield"), py::arg("unemployment_rate"))
        .def("evaluate", &gdp_engine::TaylorRuleModel::evaluate);

    py::class_<gdp_engine::InflationVetoModel>(m, "InflationVetoModel")
        .def(py::init<double, double, double>(),
             py::arg("cpi"), py::arg("core_gdp"), py::arg("nfp"))
        .def("evaluate", &gdp_engine::InflationVetoModel::evaluate);

    py::class_<gdp_engine::DollarStrengthModel>(m, "DollarStrengthModel")
        .def(py::init<double, double, double>(),
             py::arg("dxy"), py::arg("treasury_3m_pct_change"), py::arg("nfp"))
        .def("evaluate", &gdp_engine::DollarStrengthModel::evaluate);

    py::class_<gdp_engine::BondSignalModel>(m, "BondSignalModel")
        .def(py::init<double, double, double, double>(),
             py::arg("treasury_yield"), py::arg("treasury_3m_pct_change"),
             py::arg("cpi"), py::arg("core_gdp"))
        .def("evaluate", &gdp_engine::BondSignalModel::evaluate);

    // New signal models
    py::class_<gdp_engine::CreditSpreadModel>(m, "CreditSpreadModel")
        .def(py::init<double, double, double, double, double>(),
             py::arg("treasury_yield"), py::arg("treasury_3m_pct_change"),
             py::arg("unemployment_rate"), py::arg("cpi"), py::arg("core_gdp"))
        .def("evaluate", &gdp_engine::CreditSpreadModel::evaluate);

    py::class_<gdp_engine::LaborMarketModel>(m, "LaborMarketModel")
        .def(py::init<double, double, double, double>(),
             py::arg("nfp"), py::arg("unemployment_rate"),
             py::arg("consumer_sentiment"), py::arg("core_gdp"))
        .def("evaluate", &gdp_engine::LaborMarketModel::evaluate);

    py::class_<gdp_engine::GlobalRiskModel>(m, "GlobalRiskModel")
        .def(py::init<double, double, double, double>(),
             py::arg("dxy"), py::arg("consumer_sentiment"),
             py::arg("treasury_3m_pct_change"), py::arg("cpi"))
        .def("evaluate", &gdp_engine::GlobalRiskModel::evaluate);

    // ── GDPAnalyzer ──────────────────────────────────────────────
    py::class_<gdp_engine::GDPAnalyzer>(m, "GDPAnalyzer")
        .def(py::init<double, double, double, double, double, double, double, double, double, double>(),
             py::arg("cpi"), py::arg("core_gdp"), py::arg("nfp"),
             py::arg("dxy"), py::arg("treasury_3m_pct_change"),
             py::arg("treasury_yield") = 0.0,
             py::arg("unemployment_rate") = 0.0,
             py::arg("consumer_sentiment") = 0.0,
             py::arg("yield_spread_10y2y") = 0.0,
             py::arg("policy_rate") = 0.0)
        .def_property_readonly("signal", &gdp_engine::GDPAnalyzer::getSignal)
        .def_property_readonly("confidence", &gdp_engine::GDPAnalyzer::getConfidence)
        .def_property_readonly("reason", &gdp_engine::GDPAnalyzer::getReason)
        .def("get_signal", &gdp_engine::GDPAnalyzer::getSignal)
        .def("get_confidence", &gdp_engine::GDPAnalyzer::getConfidence)
        .def("get_reason", &gdp_engine::GDPAnalyzer::getReason)
        .def("get_regime", &gdp_engine::GDPAnalyzer::getRegime)
        .def("get_model_votes", &gdp_engine::GDPAnalyzer::getModelVotes)
        .def("get_ensemble_numeric_value", &gdp_engine::GDPAnalyzer::getEnsembleNumericValue)
        .def("get_methodology_summary", &gdp_engine::GDPAnalyzer::getMethodologySummary)
        .def("get_signal_strength", &gdp_engine::GDPAnalyzer::getSignalStrength)
        .def("get_dominant_model", &gdp_engine::GDPAnalyzer::getDominantModelName)
        .def("get_model_agreement", &gdp_engine::GDPAnalyzer::getModelAgreement);

    // ── Enhanced Backtest ────────────────────────────────────────
    py::class_<gdp_engine::DataPoint>(m, "DataPoint")
        .def(py::init<double, double, double, double, double, double, double, double, double>(),
             py::arg("gdp"), py::arg("cpi"), py::arg("nfp"),
             py::arg("dxy") = 0.0, py::arg("treasury_yield") = 0.0,
             py::arg("treasury_3m_change") = 0.0, py::arg("unemployment_rate") = 0.0,
             py::arg("consumer_sentiment") = 0.0, py::arg("yield_spread_10y2y") = 0.0)
        .def_readonly("gdp", &gdp_engine::DataPoint::gdp)
        .def_readonly("cpi", &gdp_engine::DataPoint::cpi)
        .def_readonly("nfp", &gdp_engine::DataPoint::nfp)
        .def_readonly("dxy", &gdp_engine::DataPoint::dxy)
        .def_readonly("treasury_yield", &gdp_engine::DataPoint::treasury_yield)
        .def_readonly("treasury_3m_change", &gdp_engine::DataPoint::treasury_3m_change)
        .def_readonly("unemployment_rate", &gdp_engine::DataPoint::unemployment_rate)
        .def_readonly("consumer_sentiment", &gdp_engine::DataPoint::consumer_sentiment)
        .def_readonly("yield_spread_10y2y", &gdp_engine::DataPoint::yield_spread_10y2y);

    py::class_<gdp_engine::TradeRecord>(m, "TradeRecord")
        .def_readonly("entry_price", &gdp_engine::TradeRecord::entry_price)
        .def_readonly("exit_price", &gdp_engine::TradeRecord::exit_price)
        .def_readonly("signal_at_entry", &gdp_engine::TradeRecord::signal_at_entry)
        .def_readonly("return_pct", &gdp_engine::TradeRecord::return_pct)
        .def_property_readonly("is_win", &gdp_engine::TradeRecord::isWin);

    py::class_<gdp_engine::VaRResult>(m, "VaRResult")
        .def_readonly("var_95", &gdp_engine::VaRResult::var_95)
        .def_readonly("var_99", &gdp_engine::VaRResult::var_99)
        .def_readonly("cvar_95", &gdp_engine::VaRResult::cvar_95)
        .def_readonly("cvar_99", &gdp_engine::VaRResult::cvar_99)
        .def_readonly("historical_var_95", &gdp_engine::VaRResult::historical_var_95)
        .def_readonly("historical_var_99", &gdp_engine::VaRResult::historical_var_99);

    py::class_<gdp_engine::EnhancedBacktestResult>(m, "EnhancedBacktestResult")
        .def_readonly("total_return_pct", &gdp_engine::EnhancedBacktestResult::total_return_pct)
        .def_readonly("annualized_return_pct", &gdp_engine::EnhancedBacktestResult::annualized_return_pct)
        .def_readonly("cumulative_return_pct", &gdp_engine::EnhancedBacktestResult::cumulative_return_pct)
        .def_readonly("volatility_pct", &gdp_engine::EnhancedBacktestResult::volatility_pct)
        .def_readonly("max_drawdown_pct", &gdp_engine::EnhancedBacktestResult::max_drawdown_pct)
        .def_readonly("downside_deviation_pct", &gdp_engine::EnhancedBacktestResult::downside_deviation_pct)
        .def_readonly("sharpe_ratio", &gdp_engine::EnhancedBacktestResult::sharpe_ratio)
        .def_readonly("sortino_ratio", &gdp_engine::EnhancedBacktestResult::sortino_ratio)
        .def_readonly("calmar_ratio", &gdp_engine::EnhancedBacktestResult::calmar_ratio)
        .def_readonly("information_ratio", &gdp_engine::EnhancedBacktestResult::information_ratio)
        .def_readonly("treynor_ratio", &gdp_engine::EnhancedBacktestResult::treynor_ratio)
        .def_readonly("beta", &gdp_engine::EnhancedBacktestResult::beta)
        .def_readonly("alpha", &gdp_engine::EnhancedBacktestResult::alpha)
        .def_readonly("var_95_pct", &gdp_engine::EnhancedBacktestResult::var_95_pct)
        .def_readonly("cvar_95_pct", &gdp_engine::EnhancedBacktestResult::cvar_95_pct)
        .def_readonly("ulcer_index", &gdp_engine::EnhancedBacktestResult::ulcer_index)
        .def_readonly("total_trades", &gdp_engine::EnhancedBacktestResult::total_trades)
        .def_readonly("winning_trades", &gdp_engine::EnhancedBacktestResult::winning_trades)
        .def_readonly("losing_trades", &gdp_engine::EnhancedBacktestResult::losing_trades)
        .def_readonly("win_rate_pct", &gdp_engine::EnhancedBacktestResult::win_rate_pct)
        .def_readonly("profit_factor", &gdp_engine::EnhancedBacktestResult::profit_factor)
        .def_readonly("avg_win_pct", &gdp_engine::EnhancedBacktestResult::avg_win_pct)
        .def_readonly("avg_loss_pct", &gdp_engine::EnhancedBacktestResult::avg_loss_pct)
        .def_readonly("largest_win_pct", &gdp_engine::EnhancedBacktestResult::largest_win_pct)
        .def_readonly("largest_loss_pct", &gdp_engine::EnhancedBacktestResult::largest_loss_pct)
        .def_readonly("avg_hold_bars", &gdp_engine::EnhancedBacktestResult::avg_hold_bars)
        .def_readonly("initial_capital", &gdp_engine::EnhancedBacktestResult::initial_capital)
        .def_readonly("final_value", &gdp_engine::EnhancedBacktestResult::final_value)
        .def_readonly("total_fees", &gdp_engine::EnhancedBacktestResult::total_fees)
        .def_readonly("equity_curve", &gdp_engine::EnhancedBacktestResult::equity_curve)
        .def_readonly("drawdown_curve", &gdp_engine::EnhancedBacktestResult::drawdown_curve)
        .def_readonly("trades", &gdp_engine::EnhancedBacktestResult::trades)
        .def("__repr__", [](const gdp_engine::EnhancedBacktestResult& r) {
            return "<BacktestResult: ret=" + std::to_string(r.total_return_pct)
                 + "%, Sharpe=" + std::to_string(r.sharpe_ratio)
                 + ", VaR95=" + std::to_string(r.var_95_pct)
                 + "%, trades=" + std::to_string(r.total_trades) + ">";
        });

    py::class_<gdp_engine::BacktestEngine>(m, "BacktestEngine")
        .def(py::init<std::vector<gdp_engine::DataPoint>, std::vector<double>>(),
             py::arg("data"), py::arg("prices"))
        .def("run_all_models", &gdp_engine::BacktestEngine::runAllModels,
             "Run ensemble backtest on all data")
        .def("run_with_model", &gdp_engine::BacktestEngine::runWithModel,
             py::arg("model_name"),
             "Run backtest using a specific signal model")
        .def("run_walk_forward", &gdp_engine::BacktestEngine::runWalkForward,
             py::arg("train_size"), py::arg("test_size"),
             "Run walk-forward validation")
        .def("run_buy_and_hold", &gdp_engine::BacktestEngine::runBuyAndHold,
             "Run buy-and-hold benchmark")
        .def("run_with_risk_controls", &gdp_engine::BacktestEngine::runWithRiskControls,
             py::arg("stop_loss_pct"), py::arg("take_profit_pct"),
             "Run backtest with stop-loss and take-profit")
        .def("run_with_position_sizing", &gdp_engine::BacktestEngine::runWithPositionSizing,
             "Run backtest with confidence-based position sizing");

    // ── Statistics Engine ─────────────────────────────────────────
    py::class_<gdp_engine::RollingStats>(m, "RollingStats")
        .def_readonly("rolling_mean", &gdp_engine::RollingStats::rolling_mean)
        .def_readonly("rolling_std", &gdp_engine::RollingStats::rolling_std)
        .def_readonly("rolling_zscore", &gdp_engine::RollingStats::rolling_zscore)
        .def_readonly("rolling_sharpe", &gdp_engine::RollingStats::rolling_sharpe);

    py::class_<gdp_engine::StatisticsEngine>(m, "StatisticsEngine")
        .def_static("mean", &gdp_engine::StatisticsEngine::mean,
                    py::arg("values"))
        .def_static("stddev", &gdp_engine::StatisticsEngine::stddev,
                    py::arg("values"))
        .def_static("correlation", &gdp_engine::StatisticsEngine::correlation,
                    py::arg("a"), py::arg("b"))
        .def_static("sharpe_ratio", &gdp_engine::StatisticsEngine::sharpeRatio,
                    py::arg("returns"), py::arg("risk_free_rate") = 0.05)
        .def_static("sortino_ratio", &gdp_engine::StatisticsEngine::sortinoRatio,
                    py::arg("returns"), py::arg("risk_free_rate") = 0.05)
        .def_static("max_drawdown", &gdp_engine::StatisticsEngine::maxDrawdown,
                    py::arg("equity"))
        .def_static("calmar_ratio", &gdp_engine::StatisticsEngine::calmarRatio,
                    py::arg("returns"), py::arg("equity"))
        .def_static("rolling_stats", &gdp_engine::StatisticsEngine::rollingStats,
                    py::arg("values"), py::arg("window"))
        .def_static("value_at_risk", &gdp_engine::StatisticsEngine::valueAtRisk,
                    py::arg("returns"), py::arg("confidence") = 0.95)
        .def_static("conditional_var", &gdp_engine::StatisticsEngine::conditionalVaR,
                    py::arg("returns"), py::arg("confidence") = 0.95)
        .def_static("historical_var", &gdp_engine::StatisticsEngine::historicalVaR,
                    py::arg("returns"), py::arg("confidence") = 0.95)
        .def_static("information_ratio", &gdp_engine::StatisticsEngine::informationRatio,
                    py::arg("returns"), py::arg("benchmark"))
        .def_static("treynor_ratio", &gdp_engine::StatisticsEngine::treynorRatio,
                    py::arg("returns"), py::arg("benchmark"),
                    py::arg("risk_free_rate") = 0.05)
        .def_static("beta", &gdp_engine::StatisticsEngine::beta,
                    py::arg("returns"), py::arg("benchmark"))
        .def_static("alpha", &gdp_engine::StatisticsEngine::alpha,
                    py::arg("returns"), py::arg("benchmark"),
                    py::arg("risk_free_rate") = 0.05)
        .def_static("ulcer_index", &gdp_engine::StatisticsEngine::ulcerIndex,
                    py::arg("equity"))
        .def_static("gain_loss_ratio", &gdp_engine::StatisticsEngine::gainLossRatio,
                    py::arg("returns"));
}