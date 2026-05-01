def generate_rule_based_report(summary: dict) -> str:
    """
    Generate a beginner-friendly stock strategy report from summary metrics.

    This function is rule-based. It does not call any AI API or external model.
    """
    initial_value = summary["initial_value"]
    final_value = summary["final_value"]
    total_return_pct = summary["total_return_pct"]
    max_drawdown_pct = summary["max_drawdown_pct"]
    buy_signals = summary["buy_signals"]
    sell_signals = summary["sell_signals"]
    currently_holding = summary["currently_holding"]

    if total_return_pct > 0:
        return_message = "The strategy made a profit during this backtest."
    elif total_return_pct < 0:
        return_message = "The strategy lost money during this backtest."
    else:
        return_message = "The strategy ended with no gain or loss during this backtest."

    if max_drawdown_pct <= -20:
        risk_message = "Risk interpretation: the drawdown was high, so the strategy may be difficult for beginners to tolerate."
    elif max_drawdown_pct <= -10:
        risk_message = "Risk interpretation: the drawdown was moderate, so risk control should be reviewed carefully."
    else:
        risk_message = "Risk interpretation: the drawdown was low in this sample, but this does not guarantee low risk in real markets."

    if currently_holding:
        holding_message = "The strategy is currently holding a position at the end of the backtest."
    else:
        holding_message = "The strategy is not holding a position at the end of the backtest."

    report = f"""
QuantPilot-AI Rule-Based Strategy Report
---------------------------------------

Initial portfolio value: {initial_value:.2f}
Final portfolio value: {final_value:.2f}
Total return: {total_return_pct:.2f}%
Maximum drawdown: {max_drawdown_pct:.2f}%

Buy signals: {buy_signals}
Sell signals: {sell_signals}

{return_message}
{holding_message}
{risk_message}

Note: This report is generated from simple rules for educational research only.
It is not financial advice.
""".strip()

    return report
