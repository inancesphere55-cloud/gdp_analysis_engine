from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class ResearchReporter:
    @staticmethod
    def generate_latex_table(df: pd.DataFrame, caption: str = "",
                              label: str = "", precision: int = 2) -> str:
        if df.empty:
            return "% Empty dataframe"

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "}",
            r"\toprule",
        ]

        header = " & ".join(df.columns) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        for _, row in df.iterrows():
            row_strs = []
            for i, val in enumerate(row):
                if isinstance(val, float):
                    row_strs.append(f"{val:.{precision}f}")
                else:
                    row_strs.append(str(val))
            lines.append(" & ".join(row_strs) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    @staticmethod
    def generate_markdown_report(analysis_result: dict, backtest_result: dict = None) -> str:
        lines = [
            "# GDP Analysis Engine — Research Report",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "---",
            "",
            "## Current Signal",
            "",
        ]

        if analysis_result:
            r = analysis_result
            lines.append(f"- **Signal:** {r.get('signal', '—')}")
            lines.append(f"- **Confidence:** {r.get('confidence', 0)}%")
            lines.append(f"- **Ensemble Value:** {r.get('ensemble_value', '—')}")
            lines.append(f"- **Macro Regime:** {r.get('regime', {}).get('label', '—')}")
            lines.append("")
            lines.append("### Regime Probabilities")
            regime = r.get("regime", {})
            for label_key in ["expansion_prob", "inflation_prob", "recession_prob",
                              "stagflation_prob", "normal_prob"]:
                label_name = label_key.replace("_prob", "").title()
                lines.append(f"- {label_name}: {regime.get(label_key, 0) * 100:.1f}%")
            lines.append("")
            lines.append("### Model Votes")
            for m in r.get("models", []):
                lines.append(f"- **{m['signal']}** (conf={m['confidence']}%, "
                           f"value={m['value']}): {m['reason']}")
            lines.append("")
            lines.append("### Analysis")
            lines.append(r.get("reason", "—"))
            lines.append("")
            lines.append("### Methodology")
            lines.append("```")
            lines.append(r.get("methodology", "—"))
            lines.append("```")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Backtest Results")
        lines.append("")

        if backtest_result:
            for key in ["strategy", "benchmark", "walk_forward"]:
                bt = backtest_result.get(key)
                if bt:
                    lines.append(f"### {bt.get('label', key)}")
                    lines.append(f"- Total Return: {bt.get('total_return_pct', 0):+.2f}%")
                    lines.append(f"- Annualized Return: {bt.get('annualized_return_pct', 0):+.2f}%")
                    lines.append(f"- Sharpe Ratio: {bt.get('sharpe_ratio', 0):.3f}")
                    lines.append(f"- Sortino Ratio: {bt.get('sortino_ratio', 0):.3f}")
                    lines.append(f"- Calmar Ratio: {bt.get('calmar_ratio', 0):.3f}")
                    lines.append(f"- Max Drawdown: {bt.get('max_drawdown_pct', 0):.2f}%")
                    lines.append(f"- Volatility: {bt.get('volatility_pct', 0):.2f}%")
                    lines.append(f"- Win Rate: {bt.get('win_rate_pct', 0):.1f}%")
                    lines.append(f"- Profit Factor: {bt.get('profit_factor', 0):.2f}")
                    lines.append(f"- Total Trades: {bt.get('total_trades', 0)}")
                    lines.append("")
            if "outperformance" in backtest_result:
                lines.append(f"**Strategy Outperformance:** {backtest_result['outperformance']:+.2f}%")
        else:
            lines.append("*No backtest data available.*")

        lines.extend([
            "",
            "---",
            "",
            "## Data Sources",
            "",
            "- **FRED API:** GDPC1, CPIAUCSL, CPILFESL, PAYEMS, UNRATE, DTWEXBGS,",
            "  DGS10, DGS2, UMCSENT, INDPRO, NAPM, FEDFUNDS, and more.",
            "",
            "## Methodology",
            "",
            "The GDP Research Engine uses an ensemble of four signal models:",
            "",
            "1. **Inflation Veto Model** — CPI threshold-based veto logic",
            "2. **Taylor Rule Model** — Policy stance estimation via Taylor (1993) rule",
            "3. **Dollar Strength Model** — DXY and yield momentum composite",
            "4. **Bond Signal Model** — Real yields and yield curve dynamics",
            "",
            "Each model votes BUY/SELL/HOLD with confidence. The ensemble",
            "combines votes weighted by confidence. Macro regime detection",
            "uses CPI, GDP, NFP, unemployment, and consumer sentiment.",
        ])

        return "\n".join(lines)

    @staticmethod
    def save_report(content: str, path: str = "data/research_report.md"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"  Report saved to: {p}")
        return p

    @staticmethod
    def csv_to_latex(csv_path: str, output_path: str = None) -> str:
        df = pd.read_csv(csv_path)
        latex = ResearchReporter.generate_latex_table(
            df, caption="Analysis Log Summary", label="tab:analysis_log"
        )
        if output_path:
            Path(output_path).write_text(latex, encoding="utf-8")
        return latex
