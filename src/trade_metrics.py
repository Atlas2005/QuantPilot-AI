import pandas as pd


def _empty_trade_metrics() -> dict:
    return {
        "total_trades": 0,
        "closed_trades": 0,
        "open_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": None,
        "total_realized_profit": 0.0,
        "average_profit": None,
        "average_loss": None,
        "average_return_pct": None,
        "best_trade_profit": None,
        "worst_trade_profit": None,
        "best_trade_return_pct": None,
        "worst_trade_return_pct": None,
        "profit_factor": None,
        "average_holding_days": None,
        "open_unrealized_profit": 0.0,
        "open_unrealized_return_pct": None,
    }


def summarize_trade_metrics(trades_df: pd.DataFrame) -> dict:
    """
    Summarize closed and open trade records from the trade log.
    """
    if trades_df.empty:
        return _empty_trade_metrics()

    metrics = _empty_trade_metrics()
    metrics["total_trades"] = int(len(trades_df))

    closed_trades = trades_df[trades_df["status"] == "closed"].copy()
    open_trades = trades_df[trades_df["status"] == "open"].copy()

    metrics["closed_trades"] = int(len(closed_trades))
    metrics["open_trades"] = int(len(open_trades))

    if not open_trades.empty:
        open_profit = pd.to_numeric(
            open_trades["unrealized_profit"],
            errors="coerce",
        ).dropna()
        open_return = pd.to_numeric(
            open_trades["unrealized_return_pct"],
            errors="coerce",
        ).dropna()

        metrics["open_unrealized_profit"] = float(open_profit.sum())
        if not open_return.empty:
            metrics["open_unrealized_return_pct"] = float(open_return.mean())

    if closed_trades.empty:
        return metrics

    profits = pd.to_numeric(closed_trades["profit"], errors="coerce").dropna()
    returns = pd.to_numeric(closed_trades["return_pct"], errors="coerce").dropna()
    holding_days = pd.to_numeric(
        closed_trades["holding_days"],
        errors="coerce",
    ).dropna()

    winning_profits = profits[profits > 0]
    losing_profits = profits[profits < 0]

    metrics["winning_trades"] = int(len(winning_profits))
    metrics["losing_trades"] = int(len(losing_profits))

    if metrics["closed_trades"] > 0:
        metrics["win_rate_pct"] = (
            metrics["winning_trades"] / metrics["closed_trades"]
        ) * 100

    if not profits.empty:
        metrics["total_realized_profit"] = float(profits.sum())
        metrics["best_trade_profit"] = float(profits.max())
        metrics["worst_trade_profit"] = float(profits.min())

    if not winning_profits.empty:
        metrics["average_profit"] = float(winning_profits.mean())

    if not losing_profits.empty:
        metrics["average_loss"] = float(losing_profits.mean())

    if not returns.empty:
        metrics["average_return_pct"] = float(returns.mean())
        metrics["best_trade_return_pct"] = float(returns.max())
        metrics["worst_trade_return_pct"] = float(returns.min())

    gross_profit = float(winning_profits.sum())
    gross_loss = abs(float(losing_profits.sum()))
    if gross_loss > 0:
        metrics["profit_factor"] = gross_profit / gross_loss

    if not holding_days.empty:
        metrics["average_holding_days"] = float(holding_days.mean())

    return metrics
