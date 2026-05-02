import pandas as pd


SUPPORTED_EXECUTION_MODES = {"same_close", "next_open", "next_close"}


def validate_execution_mode(execution_mode: str) -> None:
    if execution_mode not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(
            "execution_mode must be one of: "
            + ", ".join(sorted(SUPPORTED_EXECUTION_MODES))
        )


def calculate_commission(
    trade_value: float,
    commission_rate: float,
    min_commission: float,
) -> float:
    if trade_value <= 0:
        return 0.0

    commission = trade_value * commission_rate
    if commission_rate > 0 or min_commission > 0:
        commission = max(commission, min_commission)
    return float(commission)


def get_execution_price(row, execution_mode: str) -> float:
    if execution_mode == "next_open":
        if "open" not in row:
            raise ValueError("next_open execution requires an open column.")
        return float(row["open"])
    return float(row["close"])


def calculate_buy(
    cash: float,
    base_price: float,
    commission_rate: float,
    slippage_pct: float,
    min_commission: float,
) -> dict:
    execution_price = base_price * (1 + slippage_pct / 100)
    if execution_price <= 0 or cash <= 0:
        return {
            "shares": 0.0,
            "execution_price": execution_price,
            "gross_value": 0.0,
            "commission": 0.0,
            "slippage_cost": 0.0,
            "total_cost": 0.0,
        }

    shares = cash / execution_price
    gross_value = shares * execution_price
    commission = calculate_commission(gross_value, commission_rate, min_commission)

    if gross_value + commission > cash:
        if min_commission > 0 and commission == min_commission:
            available_cash = max(cash - min_commission, 0.0)
            shares = available_cash / execution_price
        else:
            shares = cash / (execution_price * (1 + commission_rate))

        gross_value = shares * execution_price
        commission = calculate_commission(gross_value, commission_rate, min_commission)

    total_cost = gross_value + commission
    if total_cost > cash:
        return {
            "shares": 0.0,
            "execution_price": execution_price,
            "gross_value": 0.0,
            "commission": 0.0,
            "slippage_cost": 0.0,
            "total_cost": 0.0,
        }

    slippage_cost = shares * (execution_price - base_price)
    return {
        "shares": float(shares),
        "execution_price": float(execution_price),
        "gross_value": float(gross_value),
        "commission": float(commission),
        "slippage_cost": float(slippage_cost),
        "total_cost": float(total_cost),
    }


def calculate_sell(
    shares: float,
    base_price: float,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
) -> dict:
    execution_price = base_price * (1 - slippage_pct / 100)
    gross_value = shares * execution_price
    commission = calculate_commission(gross_value, commission_rate, min_commission)
    stamp_tax = gross_value * stamp_tax_rate
    net_value = gross_value - commission - stamp_tax
    slippage_cost = shares * (base_price - execution_price)

    return {
        "execution_price": float(execution_price),
        "gross_value": float(gross_value),
        "commission": float(commission),
        "stamp_tax": float(stamp_tax),
        "slippage_cost": float(slippage_cost),
        "net_value": float(net_value),
    }


def make_daily_record(
    row,
    cash: float,
    shares: float,
    total_commission: float,
    total_stamp_tax: float,
    total_slippage_cost: float,
) -> dict:
    close_price = float(row["close"])
    position_value = shares * close_price
    total_value = cash + position_value
    total_transaction_cost = total_commission + total_stamp_tax + total_slippage_cost

    return {
        "date": row["date"],
        "close": close_price,
        "signal": row["signal"],
        "cash": cash,
        "shares": shares,
        "position_value": position_value,
        "total_value": total_value,
        "total_commission": total_commission,
        "total_stamp_tax": total_stamp_tax,
        "total_slippage_cost": total_slippage_cost,
        "total_transaction_cost": total_transaction_cost,
    }


def run_long_only_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10000,
) -> pd.DataFrame:
    """
    Run the original simple long-only backtest.

    Defaults intentionally preserve the original educational behavior:
    same-day close execution and no transaction costs.
    """
    backtest_df, _ = run_long_only_backtest_with_trades(df, initial_cash)
    return backtest_df


def run_long_only_backtest_with_trades(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    max_holding_days: int | None = None,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    slippage_pct: float = 0.0,
    min_commission: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a long-only backtest and return daily results plus a trade log.

    execution_mode controls when signals are filled:
    - same_close: execute on the current close.
    - next_open: execute on the next row's open.
    - next_close: execute on the next row's close.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must contain a signal column.")

    validate_execution_mode(execution_mode)

    result = df.copy().reset_index(drop=True)
    cash = float(initial_cash)
    shares = 0.0
    daily_records = []
    trade_records = []
    entry_date = None
    entry_price = None
    entry_total_cost = 0.0
    entry_commission = 0.0
    entry_slippage_cost = 0.0
    pending_action = None
    pending_exit_reason = None
    total_commission = 0.0
    total_stamp_tax = 0.0
    total_slippage_cost = 0.0

    def open_position(row) -> None:
        nonlocal cash, shares, entry_date, entry_price
        nonlocal entry_total_cost, entry_commission, entry_slippage_cost
        nonlocal total_commission, total_slippage_cost

        base_price = get_execution_price(row, execution_mode)
        buy = calculate_buy(
            cash=cash,
            base_price=base_price,
            commission_rate=commission_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
        )
        if buy["shares"] <= 0:
            return

        shares = buy["shares"]
        cash -= buy["total_cost"]
        entry_date = row["date"]
        entry_price = buy["execution_price"]
        entry_total_cost = buy["total_cost"]
        entry_commission = buy["commission"]
        entry_slippage_cost = buy["slippage_cost"]
        total_commission += buy["commission"]
        total_slippage_cost += buy["slippage_cost"]

    def close_position(row, exit_reason: str) -> None:
        nonlocal cash, shares, entry_date, entry_price, entry_total_cost
        nonlocal entry_commission, entry_slippage_cost
        nonlocal total_commission, total_stamp_tax, total_slippage_cost

        base_price = get_execution_price(row, execution_mode)
        sell = calculate_sell(
            shares=shares,
            base_price=base_price,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
        )
        cash += sell["net_value"]
        total_commission += sell["commission"]
        total_stamp_tax += sell["stamp_tax"]
        total_slippage_cost += sell["slippage_cost"]

        profit = sell["net_value"] - entry_total_cost
        return_pct = (
            None
            if entry_total_cost == 0
            else (profit / entry_total_cost) * 100
        )
        holding_days = (
            pd.to_datetime(row["date"]) - pd.to_datetime(entry_date)
        ).days

        trade_records.append(
            {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": row["date"],
                "exit_price": sell["execution_price"],
                "shares": shares,
                "profit": profit,
                "return_pct": return_pct,
                "unrealized_profit": None,
                "unrealized_return_pct": None,
                "holding_days": holding_days,
                "status": "closed",
                "exit_reason": exit_reason,
                "entry_commission": entry_commission,
                "exit_commission": sell["commission"],
                "stamp_tax": sell["stamp_tax"],
                "slippage_cost": entry_slippage_cost + sell["slippage_cost"],
                "total_transaction_cost": (
                    entry_commission
                    + sell["commission"]
                    + sell["stamp_tax"]
                    + entry_slippage_cost
                    + sell["slippage_cost"]
                ),
            }
        )

        shares = 0.0
        entry_date = None
        entry_price = None
        entry_total_cost = 0.0
        entry_commission = 0.0
        entry_slippage_cost = 0.0

    for index, row in result.iterrows():
        close_price = float(row["close"])
        signal = row["signal"]

        if pending_action == "buy" and shares == 0:
            open_position(row)
        elif pending_action == "sell" and shares > 0:
            close_position(row, pending_exit_reason or "signal")
        pending_action = None
        pending_exit_reason = None

        if shares > 0:
            current_return_pct = ((close_price - entry_price) / entry_price) * 100
            holding_days = (
                pd.to_datetime(row["date"]) - pd.to_datetime(entry_date)
            ).days
            exit_reason = None

            if stop_loss_pct is not None and current_return_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif take_profit_pct is not None and current_return_pct >= take_profit_pct:
                exit_reason = "take_profit"
            elif max_holding_days is not None and holding_days >= max_holding_days:
                exit_reason = "max_holding_days"
            elif signal == -1:
                exit_reason = "signal"

            if exit_reason is not None:
                if execution_mode == "same_close":
                    close_position(row, exit_reason)
                elif index < len(result) - 1:
                    pending_action = "sell"
                    pending_exit_reason = exit_reason

        elif signal == 1:
            if execution_mode == "same_close":
                open_position(row)
            elif index < len(result) - 1:
                pending_action = "buy"

        daily_records.append(
            make_daily_record(
                row=row,
                cash=cash,
                shares=shares,
                total_commission=total_commission,
                total_stamp_tax=total_stamp_tax,
                total_slippage_cost=total_slippage_cost,
            )
        )

    if shares > 0:
        final_row = result.iloc[-1]
        final_date = final_row["date"]
        final_price = float(final_row["close"])
        current_value = shares * final_price
        unrealized_profit = current_value - entry_total_cost
        unrealized_return_pct = (
            None
            if entry_total_cost == 0
            else (unrealized_profit / entry_total_cost) * 100
        )
        holding_days = (
            pd.to_datetime(final_date) - pd.to_datetime(entry_date)
        ).days

        trade_records.append(
            {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": None,
                "exit_price": None,
                "shares": shares,
                "profit": None,
                "return_pct": None,
                "unrealized_profit": unrealized_profit,
                "unrealized_return_pct": unrealized_return_pct,
                "holding_days": holding_days,
                "status": "open",
                "exit_reason": "open",
                "entry_commission": entry_commission,
                "exit_commission": None,
                "stamp_tax": None,
                "slippage_cost": entry_slippage_cost,
                "total_transaction_cost": entry_commission + entry_slippage_cost,
            }
        )

    return pd.DataFrame(daily_records), pd.DataFrame(trade_records)
