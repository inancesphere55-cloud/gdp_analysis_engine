#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <numeric>
#include "gdp_engine.hpp"

namespace gdp_engine {

// ── Regime Detection ────────────────────────────────────────────────

std::string regimeToString(MacroRegime r) {
    switch (r) {
        case MacroRegime::EXPANSION:    return "EXPANSION";
        case MacroRegime::INFLATIONARY: return "INFLATIONARY";
        case MacroRegime::RECESSION:    return "RECESSION";
        case MacroRegime::STAGFLATION:  return "STAGFLATION";
        case MacroRegime::NORMAL:       return "NORMAL";
        default:                        return "UNKNOWN";
    }
}

RegimeDetector::RegimeDetector(double cpi, double core_gdp, double nfp,
                               double dxy, double treasury_yield,
                               double unemployment_rate, double consumer_sentiment,
                               double yield_spread_10y2y)
    : m_cpi(cpi), m_core_gdp(core_gdp), m_nfp(nfp), m_dxy(dxy)
    , m_treasury_yield(treasury_yield), m_unemployment_rate(unemployment_rate)
    , m_consumer_sentiment(consumer_sentiment), m_yield_spread(yield_spread_10y2y) {}

RegimeResult RegimeDetector::analyze() const {
    RegimeResult result{};

    bool high_inflation = m_cpi > 3.0;
    bool low_growth = m_core_gdp < 1.0;
    bool moderate_growth = m_core_gdp >= 1.0 && m_core_gdp <= 2.5;
    bool high_growth = m_core_gdp > 2.5;
    bool weak_labor = m_nfp < 100000 || m_unemployment_rate > 6.0;
    bool strong_labor = m_nfp > 200000 && m_unemployment_rate < 4.5;
    bool strong_dollar = m_dxy > 105.0;
    bool inverted_curve = m_yield_spread < -0.4;
    bool bear_steepener = m_treasury_yield > 4.5 && m_cpi > 3.0;
    bool weak_sentiment = m_consumer_sentiment < 60.0;
    bool strong_sentiment = m_consumer_sentiment > 85.0;

    double score = 0.0;
    score += (high_growth ? 1.0 : low_growth ? -1.0 : 0.0);
    score += (high_inflation ? -1.5 : 0.0);
    score += (weak_labor ? -1.0 : strong_labor ? 1.0 : 0.0);
    score += (inverted_curve ? -1.5 : 0.0);
    score += (weak_sentiment ? -0.5 : strong_sentiment ? 0.5 : 0.0);

    if (inverted_curve && weak_sentiment && low_growth) {
        result.regime = MacroRegime::RECESSION;
        result.recession_prob = 0.70;
        result.stagflation_prob = 0.12;
        result.normal_prob = 0.08;
        result.expansion_prob = 0.05;
        result.inflation_prob = 0.05;
    } else if (high_inflation && low_growth) {
        result.regime = MacroRegime::STAGFLATION;
        result.stagflation_prob = 0.65;
        result.inflation_prob = 0.20;
        result.recession_prob = 0.10;
        result.expansion_prob = 0.02;
        result.normal_prob = 0.03;
    } else if (high_inflation && high_growth) {
        result.regime = MacroRegime::INFLATIONARY;
        result.inflation_prob = 0.60;
        result.expansion_prob = 0.25;
        result.stagflation_prob = 0.08;
        result.recession_prob = 0.02;
        result.normal_prob = 0.05;
    } else if (low_growth && weak_labor) {
        result.regime = MacroRegime::RECESSION;
        result.recession_prob = 0.55;
        result.stagflation_prob = 0.15;
        result.normal_prob = 0.15;
        result.expansion_prob = 0.10;
        result.inflation_prob = 0.05;
    } else if (moderate_growth && strong_labor && !high_inflation) {
        result.regime = MacroRegime::EXPANSION;
        result.expansion_prob = 0.55;
        result.normal_prob = 0.30;
        result.inflation_prob = 0.10;
        result.recession_prob = 0.03;
        result.stagflation_prob = 0.02;
    } else {
        result.regime = MacroRegime::NORMAL;
        result.normal_prob = 0.50;
        result.expansion_prob = 0.20;
        result.inflation_prob = 0.15;
        result.recession_prob = 0.10;
        result.stagflation_prob = 0.05;
    }

    return result;
}

// ── Taylor Rule Model ───────────────────────────────────────────────

TaylorRuleModel::TaylorRuleModel(double cpi, double core_gdp, double treasury_yield,
                                 double unemployment_rate)
    : m_cpi(cpi), m_core_gdp(core_gdp), m_treasury_yield(treasury_yield)
    , m_unemployment_rate(unemployment_rate) {}

SignalVote TaylorRuleModel::evaluate() const {
    double real_rate = m_treasury_yield - m_cpi;
    double neutral_rate = 2.0;
    double gdp_gap = m_core_gdp - 1.8;
    double implied_rate = neutral_rate + (m_cpi - 2.0) + gdp_gap;
    double policy_stance = real_rate - implied_rate;

    SignalVote vote{};
    vote.numeric_value = 0.0;

    if (policy_stance < -2.0) {
        vote.signal = "STRONG_BUY";
        vote.confidence = 85;
        vote.numeric_value = 2.0;
        vote.reason = "Taylor rule implies deeply accommodative policy (gap="
                      + std::to_string(policy_stance) + ")";
    } else if (policy_stance < -0.5) {
        vote.signal = "BUY";
        vote.confidence = 70;
        vote.numeric_value = 1.0;
        vote.reason = "Taylor rule suggests accommodative policy (gap="
                      + std::to_string(policy_stance) + ")";
    } else if (policy_stance > 2.0) {
        vote.signal = "STRONG_SELL";
        vote.confidence = 85;
        vote.numeric_value = -2.0;
        vote.reason = "Taylor rule implies overly restrictive policy (gap="
                      + std::to_string(policy_stance) + ")";
    } else if (policy_stance > 0.5) {
        vote.signal = "SELL";
        vote.confidence = 70;
        vote.numeric_value = -1.0;
        vote.reason = "Taylor rule suggests restrictive policy (gap="
                      + std::to_string(policy_stance) + ")";
    } else {
        vote.signal = "HOLD";
        vote.confidence = 60;
        vote.numeric_value = 0.0;
        vote.reason = "Taylor rule near neutral (gap="
                      + std::to_string(policy_stance) + ")";
    }

    return vote;
}

// ── Inflation Veto Model ────────────────────────────────────────────

InflationVetoModel::InflationVetoModel(double cpi, double core_gdp, double nfp)
    : m_cpi(cpi), m_core_gdp(core_gdp), m_nfp(nfp) {}

SignalVote InflationVetoModel::evaluate() const {
    SignalVote vote{};

    if (m_cpi > 4.5) {
        vote.signal = "STRONG_SELL";
        vote.confidence = 95;
        vote.numeric_value = -2.0;
        vote.reason = "CPI at " + std::to_string(m_cpi) + "% - critical inflation veto";
    } else if (m_cpi > 3.5) {
        vote.signal = "SELL";
        vote.confidence = 80;
        vote.numeric_value = -1.5;
        vote.reason = "CPI at " + std::to_string(m_cpi) + "% - inflation veto triggered";
    } else if (m_cpi > 2.5) {
        vote.signal = "HOLD";
        vote.confidence = 55;
        vote.numeric_value = -0.5;
        vote.reason = "CPI at " + std::to_string(m_cpi) + "% - elevated but not critical";
    } else if (m_core_gdp < 1.0 && m_nfp < 150000) {
        vote.signal = "STRONG_BUY";
        vote.confidence = 85;
        vote.numeric_value = 2.0;
        vote.reason = "Low CPI (" + std::to_string(m_cpi) + "%) + weak GDP + weak labor";
    } else if (m_core_gdp < 1.5) {
        vote.signal = "BUY";
        vote.confidence = 70;
        vote.numeric_value = 1.0;
        vote.reason = "Controlled CPI (" + std::to_string(m_cpi) + "%) + weak GDP";
    } else {
        vote.signal = "HOLD";
        vote.confidence = 50;
        vote.numeric_value = 0.0;
        vote.reason = "CPI controlled at " + std::to_string(m_cpi) + "%";
    }

    return vote;
}

// ── Dollar Strength Model ───────────────────────────────────────────

DollarStrengthModel::DollarStrengthModel(double dxy, double treasury_3m_pct_change, double nfp)
    : m_dxy(dxy), m_treasury_3m(treasury_3m_pct_change), m_nfp(nfp) {}

SignalVote DollarStrengthModel::evaluate() const {
    SignalVote vote{};

    double dxy_strength = (m_dxy - 100.0) / 10.0;
    double bond_rally = -m_treasury_3m / 10.0;
    double composite = dxy_strength + bond_rally;

    if (composite > 1.5) {
        vote.signal = "STRONG_SELL";
        vote.confidence = 85;
        vote.numeric_value = -2.0;
        vote.reason = "DXY very strong (" + std::to_string(m_dxy) + "), bearish risk assets";
    } else if (composite > 0.5) {
        vote.signal = "SELL";
        vote.confidence = 70;
        vote.numeric_value = -1.0;
        vote.reason = "Dollar strength pressuring risk (DXY=" + std::to_string(m_dxy) + ")";
    } else if (composite < -1.0) {
        vote.signal = "BUY";
        vote.confidence = 70;
        vote.numeric_value = 1.0;
        vote.reason = "Weak dollar supportive for risk (DXY=" + std::to_string(m_dxy) + ")";
    } else {
        vote.signal = "HOLD";
        vote.confidence = 50;
        vote.numeric_value = 0.0;
        vote.reason = "Dollar neutral (DXY=" + std::to_string(m_dxy) + ")";
    }

    return vote;
}

// ── Bond Signal Model ───────────────────────────────────────────────

BondSignalModel::BondSignalModel(double treasury_yield, double treasury_3m_pct_change,
                                 double cpi, double core_gdp)
    : m_treasury_yield(treasury_yield), m_treasury_3m(treasury_3m_pct_change)
    , m_cpi(cpi), m_core_gdp(core_gdp) {}

SignalVote BondSignalModel::evaluate() const {
    SignalVote vote{};

    double real_yield = m_treasury_yield - m_cpi;
    double yield_momentum = m_treasury_3m;

    if (yield_momentum > 25.0 && real_yield < -1.0) {
        vote.signal = "STRONG_SELL";
        vote.confidence = 90;
        vote.numeric_value = -2.0;
        vote.reason = "Bond selloff accelerating + deeply negative real yields = stagflation fear";
    } else if (yield_momentum > 15.0) {
        vote.signal = "SELL";
        vote.confidence = 75;
        vote.numeric_value = -1.5;
        vote.reason = "Rapid yield rise (" + std::to_string(yield_momentum) + "% in 3mo) - bond vigilantes";
    } else if (real_yield > 1.5 && yield_momentum < -5.0) {
        vote.signal = "BUY";
        vote.confidence = 70;
        vote.numeric_value = 1.0;
        vote.reason = "Positive real yields + falling yields = flight to quality, deflation signal";
    } else {
        vote.signal = "HOLD";
        vote.confidence = 50;
        vote.numeric_value = 0.0;
        vote.reason = "Bond market stable (10Y=" + std::to_string(m_treasury_yield)
                      + "%, 3m chg=" + std::to_string(yield_momentum) + "%)";
    }

    return vote;
}

// ── Credit Spread Model (NEW) ────────────────────────────────────────

CreditSpreadModel::CreditSpreadModel(double treasury_yield, double treasury_3m_pct_change,
                                     double unemployment_rate, double cpi, double core_gdp)
    : m_treasury_yield(treasury_yield), m_treasury_3m(treasury_3m_pct_change)
    , m_unemployment_rate(unemployment_rate), m_cpi(cpi), m_core_gdp(core_gdp) {}

SignalVote CreditSpreadModel::evaluate() const {
    SignalVote vote{};
    double real_yield = m_treasury_yield - m_cpi;
    double labor_stress = m_unemployment_rate > 5.5 ? (m_unemployment_rate - 5.5) * 2.0 : 0.0;
    double growth_stress = m_core_gdp < 1.0 ? (1.0 - m_core_gdp) * 2.0 : 0.0;
    double credit_stress = labor_stress + growth_stress;

    if (real_yield < -2.0 && credit_stress > 2.0) {
        vote.signal = "STRONG_SELL";
        vote.confidence = 90;
        vote.numeric_value = -2.0;
        vote.reason = "Credit stress elevated: negative real yields + labor/growth weakness";
    } else if (real_yield < -1.0 || credit_stress > 1.5) {
        vote.signal = "SELL";
        vote.confidence = 75;
        vote.numeric_value = -1.5;
        vote.reason = "Credit conditions deteriorating (stress=" + std::to_string(credit_stress) + ")";
    } else if (real_yield > 1.0 && credit_stress < 0.5) {
        vote.signal = "BUY";
        vote.confidence = 70;
        vote.numeric_value = 1.0;
        vote.reason = "Healthy credit conditions: positive real yields, low stress";
    } else {
        vote.signal = "HOLD";
        vote.confidence = 55;
        vote.numeric_value = 0.0;
        vote.reason = "Credit conditions neutral (stress=" + std::to_string(credit_stress) + ")";
    }

    return vote;
}

// ── Labor Market Model (NEW) ─────────────────────────────────────────

LaborMarketModel::LaborMarketModel(double nfp, double unemployment_rate,
                                   double consumer_sentiment, double core_gdp)
    : m_nfp(nfp), m_unemployment_rate(unemployment_rate)
    , m_consumer_sentiment(consumer_sentiment), m_core_gdp(core_gdp) {}

SignalVote LaborMarketModel::evaluate() const {
    SignalVote vote{};

    double labor_health = 0.0;
    if (m_nfp > 250000) labor_health += 2.0;
    else if (m_nfp > 150000) labor_health += 1.0;
    else if (m_nfp > 50000) labor_health += 0.0;
    else labor_health -= 1.5;

    if (m_unemployment_rate < 4.0) labor_health += 1.5;
    else if (m_unemployment_rate < 5.0) labor_health += 0.5;
    else if (m_unemployment_rate < 6.0) labor_health -= 0.5;
    else labor_health -= 2.0;

    if (m_consumer_sentiment > 80) labor_health += 1.0;
    else if (m_consumer_sentiment < 60) labor_health -= 1.0;

    if (labor_health >= 3.0) {
        vote.signal = "SELL";
        vote.confidence = 75;
        vote.numeric_value = -1.0;
        vote.reason = "Overheated labor market: tight conditions fuel wage inflation";
    } else if (labor_health >= 1.5) {
        vote.signal = "HOLD";
        vote.confidence = 60;
        vote.numeric_value = -0.3;
        vote.reason = "Healthy labor market, neutral for policy";
    } else if (labor_health >= 0.0) {
        vote.signal = "BUY";
        vote.confidence = 65;
        vote.numeric_value = 0.7;
        vote.reason = "Soft labor market supports accommodative policy";
    } else {
        vote.signal = "STRONG_BUY";
        vote.confidence = 85;
        vote.numeric_value = 1.5;
        vote.reason = "Weak labor market demands urgent policy support";
    }

    return vote;
}

// ── Global Risk Model (NEW) ──────────────────────────────────────────

GlobalRiskModel::GlobalRiskModel(double dxy, double consumer_sentiment,
                                 double treasury_3m_pct_change, double cpi)
    : m_dxy(dxy), m_consumer_sentiment(consumer_sentiment)
    , m_treasury_3m(treasury_3m_pct_change), m_cpi(cpi) {}

SignalVote GlobalRiskModel::evaluate() const {
    SignalVote vote{};

    double risk_score = 0.0;
    risk_score += (m_dxy > 110.0) ? -1.5 : (m_dxy > 103.0) ? -0.5 : 0.5;
    risk_score += (m_consumer_sentiment < 60.0) ? -1.0 : (m_consumer_sentiment > 85.0) ? 0.5 : 0.0;
    risk_score += (m_treasury_3m > 20.0) ? -1.0 : (m_treasury_3m > 10.0) ? -0.5 : 0.0;
    risk_score += (m_cpi > 4.0) ? -1.0 : 0.0;

    if (risk_score <= -2.0) {
        vote.signal = "STRONG_SELL";
        vote.confidence = 90;
        vote.numeric_value = -2.0;
        vote.reason = "High risk aversion: strong dollar, weak sentiment, rising yields";
    } else if (risk_score <= -0.5) {
        vote.signal = "SELL";
        vote.confidence = 70;
        vote.numeric_value = -1.0;
        vote.reason = "Elevated risk aversion (risk_score=" + std::to_string(risk_score) + ")";
    } else if (risk_score >= 1.0) {
        vote.signal = "BUY";
        vote.confidence = 70;
        vote.numeric_value = 1.0;
        vote.reason = "Risk-on environment: weak dollar, strong sentiment";
    } else {
        vote.signal = "HOLD";
        vote.confidence = 50;
        vote.numeric_value = 0.0;
        vote.reason = "Risk sentiment neutral (risk_score=" + std::to_string(risk_score) + ")";
    }

    return vote;
}

// ── GDP Analyzer ────────────────────────────────────────────────────

GDPAnalyzer::GDPAnalyzer(double cpi, double core_gdp, double nfp,
                         double dxy, double treasury_3m_pct_change,
                         double treasury_yield, double unemployment_rate,
                         double consumer_sentiment, double yield_spread_10y2y)
    : m_cpi(cpi), m_core_gdp(core_gdp), m_nfp(nfp), m_dxy(dxy)
    , m_treasury_3m(treasury_3m_pct_change), m_treasury_yield(treasury_yield)
    , m_unemployment_rate(unemployment_rate)
    , m_consumer_sentiment(consumer_sentiment), m_yield_spread(yield_spread_10y2y) {}

RegimeResult GDPAnalyzer::getRegime() const {
    RegimeDetector detector(m_cpi, m_core_gdp, m_nfp, m_dxy, m_treasury_yield,
                            m_unemployment_rate, m_consumer_sentiment, m_yield_spread);
    return detector.analyze();
}

std::vector<SignalVote> GDPAnalyzer::getModelVotes() const {
    std::vector<SignalVote> votes;

    InflationVetoModel veto(m_cpi, m_core_gdp, m_nfp);
    votes.push_back(veto.evaluate());

    TaylorRuleModel taylor(m_cpi, m_core_gdp, m_treasury_yield, m_unemployment_rate);
    votes.push_back(taylor.evaluate());

    DollarStrengthModel dollar(m_dxy, m_treasury_3m, m_nfp);
    votes.push_back(dollar.evaluate());

    BondSignalModel bond(m_treasury_yield, m_treasury_3m, m_cpi, m_core_gdp);
    votes.push_back(bond.evaluate());

    CreditSpreadModel credit(m_treasury_yield, m_treasury_3m,
                             m_unemployment_rate, m_cpi, m_core_gdp);
    votes.push_back(credit.evaluate());

    LaborMarketModel labor(m_nfp, m_unemployment_rate,
                           m_consumer_sentiment, m_core_gdp);
    votes.push_back(labor.evaluate());

    GlobalRiskModel risk(m_dxy, m_consumer_sentiment, m_treasury_3m, m_cpi);
    votes.push_back(risk.evaluate());

    return votes;
}

double GDPAnalyzer::getEnsembleNumericValue() const {
    auto votes = getModelVotes();
    double sum = 0.0;
    double weight_sum = 0.0;

    for (const auto& v : votes) {
        double w = static_cast<double>(v.confidence) / 100.0;
        sum += v.numeric_value * w;
        weight_sum += w;
    }

    return weight_sum > 0.0 ? sum / weight_sum : 0.0;
}

double GDPAnalyzer::getSignalStrength() const {
    return std::abs(getEnsembleNumericValue()) / 2.0;
}

std::string GDPAnalyzer::getDominantModelName() const {
    auto votes = getModelVotes();
    if (votes.empty()) return "NONE";

    size_t best_idx = 0;
    double best_abs = std::abs(votes[0].numeric_value);
    for (size_t i = 1; i < votes.size(); ++i) {
        double abs_val = std::abs(votes[i].numeric_value);
        if (abs_val > best_abs) {
            best_abs = abs_val;
            best_idx = i;
        }
    }

    static const char* names[] = {
        "InflationVeto", "TaylorRule", "DollarStrength",
        "BondSignal", "CreditSpread", "LaborMarket", "GlobalRisk"
    };
    return best_idx < 7 ? names[best_idx] : "UNKNOWN";
}

double GDPAnalyzer::getModelAgreement() const {
    auto votes = getModelVotes();
    if (votes.empty()) return 0.0;

    int buy = 0, sell = 0, hold = 0;
    for (const auto& v : votes) {
        if (v.numeric_value > 0.3) ++buy;
        else if (v.numeric_value < -0.3) ++sell;
        else ++hold;
    }

    int max_votes = std::max({buy, sell, hold});
    return static_cast<double>(max_votes) / votes.size();
}

std::string GDPAnalyzer::getMethodologySummary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    auto votes = getModelVotes();
    auto regime = getRegime();

    ss << "ENSEMBLE METHODOLOGY (7 models)\n";
    ss << "Regime: " << regimeToString(regime.regime) << "\n";
    ss << "Model agreement: " << (getModelAgreement() * 100.0) << "%\n\n";

    static const char* names[] = {
        "InflationVeto", "TaylorRule", "DollarStrength",
        "BondSignal", "CreditSpread", "LaborMarket", "GlobalRisk"
    };

    for (size_t i = 0; i < votes.size(); ++i) {
        ss << names[i] << ": " << votes[i].signal
           << " (conf=" << votes[i].confidence
           << "%, val=" << votes[i].numeric_value << ")\n"
           << "  " << votes[i].reason << "\n";
    }

    ss << "\nEnsemble value: " << getEnsembleNumericValue();
    ss << " | Strength: " << (getSignalStrength() * 100.0) << "%";

    return ss.str();
}

std::string GDPAnalyzer::getSignal() const {
    double ensemble_val = getEnsembleNumericValue();

    if (ensemble_val <= -1.5) return "STRONG_SELL";
    if (ensemble_val <= -0.5) return "SELL";
    if (ensemble_val >= 1.5) return "STRONG_BUY";
    if (ensemble_val >= 0.5) return "BUY";
    return "HOLD";
}

int GDPAnalyzer::getConfidence() const {
    auto votes = getModelVotes();
    int total_conf = 0;
    int count = 0;

    for (const auto& v : votes) {
        total_conf += v.confidence;
        ++count;
    }

    int base = count > 0 ? total_conf / count : 50;
    double agreement = getModelAgreement();
    int adjusted = static_cast<int>(base * (0.5 + 0.5 * agreement));
    return std::clamp(adjusted, 0, 100);
}

std::string GDPAnalyzer::getReason() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    auto votes = getModelVotes();
    auto regime = getRegime();

    ss << "Regime: " << regimeToString(regime.regime) << ". ";
    ss << "Ensemble signal: " << getSignal()
       << " (value=" << getEnsembleNumericValue()
       << ", strength=" << (getSignalStrength() * 100.0)
       << "%, agreement=" << (getModelAgreement() * 100.0) << "%). ";

    static const char* names[] = {
        "InflationVeto", "TaylorRule", "DollarStrength",
        "BondSignal", "CreditSpread", "LaborMarket", "GlobalRisk"
    };

    for (size_t i = 0; i < votes.size(); ++i) {
        ss << names[i] << ": " << votes[i].reason << " ";
    }

    ss << "Overall confidence: " << getConfidence() << "%.";
    return ss.str();
}

} // namespace gdp_engine