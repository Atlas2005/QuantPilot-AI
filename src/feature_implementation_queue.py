from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .feature_source_registry import get_all_features
except ImportError:
    from feature_source_registry import get_all_features


PRIORITY_STAGE = {
    "P0": "P0_now",
    "P1": "P1_next",
    "P2": "P2_later",
    "P3": "P3_research_only",
}
PRIORITY_SCORE = {"P0": 100, "P1": 70, "P2": 40, "P3": 10}
TRAINING_VALUE_SCORE = {"high": 30, "medium": 15, "low": 5, "experimental": 0}
LEAKAGE_SCORE = {"low": 20, "medium": 5, "high": -30}
COST_SCORE = {"free": 15, "low": 8, "medium": 0, "high": -15, "unknown": -5}
TOKEN_SCORE = {False: 10, True: -5}
DIFFICULTY_SCORE = {"low": 15, "medium": 5, "high": -10}
LEAKAGE_SORT = {"low": 0, "medium": 1, "high": 2}
STAGE_SORT = {"P0_now": 0, "P1_next": 1, "P2_later": 2, "P3_research_only": 3}


@dataclass
class FeatureQueueItem:
    queue_id: str
    priority: str
    category: str
    feature_group: str
    feature_name: str
    source_name: str
    data_frequency: str
    training_value: str
    implementation_stage: str
    implementation_difficulty: str
    leakage_risk: str
    token_required: bool
    cost_level: str
    recommended_action: str
    output_columns: list[str]
    validation_checks: list[str]
    notes: str
    implementation_score: int


QUEUE_DETAILS = {
    "price_action": {
        "category": "price_and_technical",
        "difficulty": "low",
        "action": "Extend the existing factor builder with longer-horizon return, MA distance, slope, momentum, RSI, and CCI features using trailing windows only.",
        "outputs": [
            "return_60d",
            "gap_return",
            "intraday_reversal",
            "ma120_gap_pct",
            "ma250_gap_pct",
            "trend_slope_20d",
            "momentum_acceleration",
            "rsi_14",
            "cci_20",
        ],
        "checks": [
            "Confirm all rolling calculations use current and past rows only.",
            "Run missing-value checks after long rolling windows.",
            "Verify future_return and label columns are excluded from features.",
        ],
    },
    "volume_liquidity": {
        "category": "price_and_technical",
        "difficulty": "low",
        "action": "Add turnover, amount, abnormal volume, dry-up, and volume-price divergence features from token-free daily market fields where available.",
        "outputs": [
            "turnover_rate",
            "amount",
            "volume_change_20d",
            "abnormal_volume_20d",
            "liquidity_dry_up",
            "volume_price_divergence",
        ],
        "checks": [
            "Check zero-volume and suspension rows.",
            "Validate amount and turnover units by source.",
            "Confirm liquidity filters do not remove future rows based on labels.",
        ],
    },
    "volatility_risk": {
        "category": "price_and_technical",
        "difficulty": "low",
        "action": "Add trailing volatility, drawdown, ATR, downside volatility, and benchmark beta features before adding external risk data.",
        "outputs": [
            "volatility_60d",
            "downside_volatility_20d",
            "atr_14",
            "max_drawdown_60d",
            "beta_to_index_60d",
            "tail_risk_60d",
        ],
        "checks": [
            "Use trailing windows only.",
            "Align benchmark dates before beta calculation.",
            "Check for division by zero and inf values.",
        ],
    },
    "valuation": {
        "category": "valuation",
        "difficulty": "medium",
        "action": "Prototype token-free valuation joins with PE, PB, PS, PCF, EV/EBITDA, dividend yield, percentiles, PEG, and ROE-to-valuation matching.",
        "outputs": [
            "pe_ttm",
            "pb",
            "ps",
            "pcf",
            "ev_ebitda",
            "dividend_yield",
            "valuation_percentile_3y",
            "industry_relative_pe",
            "historical_valuation_range",
            "peg",
            "roe_to_pb_match",
        ],
        "checks": [
            "Join by data availability date, not later restated values.",
            "Winsorize extreme valuation ratios.",
            "Compare own-history and industry-relative distributions.",
        ],
    },
    "fundamentals_profitability": {
        "category": "fundamental",
        "difficulty": "medium",
        "action": "Add profitability and quality fields with report-announcement-date lagging before using analyst revisions.",
        "outputs": [
            "gross_margin",
            "operating_margin",
            "net_margin",
            "roe",
            "roa",
            "roic",
            "earnings_surprise",
            "analyst_revision",
        ],
        "checks": [
            "Use report publication date instead of fiscal period end.",
            "Confirm no future quarterly data is forward-filled backward.",
            "Flag missing or single-quarter values.",
        ],
    },
    "fundamentals_growth": {
        "category": "fundamental",
        "difficulty": "medium",
        "action": "Add lag-safe growth and acceleration features from quarterly statements.",
        "outputs": [
            "revenue_growth_yoy",
            "revenue_growth_qoq",
            "net_profit_growth_yoy",
            "net_profit_growth_qoq",
            "eps_growth",
            "cash_flow_growth",
            "growth_acceleration",
        ],
        "checks": [
            "Use announced statements only.",
            "Handle negative base periods carefully.",
            "Check growth outliers and restatements.",
        ],
    },
    "balance_sheet_quality": {
        "category": "fundamental",
        "difficulty": "medium",
        "action": "Add balance-sheet risk and leverage features after profitability and growth fields are stable.",
        "outputs": [
            "debt_ratio",
            "current_ratio",
            "quick_ratio",
            "interest_coverage",
            "goodwill_ratio",
            "inventory_growth",
            "receivable_growth",
            "leverage_change",
        ],
        "checks": [
            "Lag statements by announcement date.",
            "Check for accounting denominator edge cases.",
            "Flag extreme goodwill and leverage values.",
        ],
    },
    "cash_flow_quality": {
        "category": "fundamental",
        "difficulty": "medium",
        "action": "Add cash-flow quality features to test whether profits are cash-backed.",
        "outputs": [
            "operating_cash_flow",
            "free_cash_flow",
            "ocf_to_net_profit",
            "capex",
            "cash_conversion_ratio",
            "cash_flow_stability",
        ],
        "checks": [
            "Lag by statement announcement date.",
            "Do not backward-fill future quarterly cash flow.",
            "Check capex and free-cash-flow sign conventions.",
        ],
    },
    "dividend_shareholder_return": {
        "category": "valuation",
        "difficulty": "medium",
        "action": "Add dividend yield, payout, stability, and shareholder return fields only from announced corporate actions.",
        "outputs": [
            "cash_dividend",
            "dividend_yield",
            "payout_ratio",
            "dividend_stability",
            "buyback_amount",
            "shareholder_return_score",
        ],
        "checks": [
            "Use announcement and ex-dividend dates correctly.",
            "Avoid using approved payouts before announcement.",
            "Check high dividend yield for price-collapse artifacts.",
        ],
    },
    "fund_holdings": {
        "category": "fund_and_institutional",
        "difficulty": "high",
        "action": "Delay until disclosure-date holdings can be collected reliably; then add fund concentration, position change, and crowding features.",
        "outputs": [
            "active_fund_holding_ratio",
            "passive_fund_holding_ratio",
            "public_fund_ranking",
            "fund_position_change",
            "fund_crowding_score",
            "top_holder_concentration",
        ],
        "checks": [
            "Use disclosure date, not report date.",
            "Separate active and passive funds.",
            "Check stale quarterly holdings coverage.",
        ],
    },
    "foreign_institutional_flow": {
        "category": "capital_flow",
        "difficulty": "medium",
        "action": "Prototype northbound and foreign holding features from sources with clear timestamps.",
        "outputs": [
            "northbound_net_flow",
            "foreign_holding_ratio",
            "foreign_holding_change",
            "foreign_ownership_percentile",
            "foreign_inflow_acceleration",
        ],
        "checks": [
            "Validate timestamp availability before prediction time.",
            "Check source coverage by stock.",
            "Compare flow persistence across windows.",
        ],
    },
    "stabilization_or_policy_funds": {
        "category": "fund_and_institutional",
        "difficulty": "high",
        "action": "Keep as research-only until source definitions and point-in-time disclosures are auditable.",
        "outputs": [
            "stabilization_fund_holding_ratio",
            "stabilization_fund_holding_change",
            "policy_support_signal",
        ],
        "checks": [
            "Require explicit event or disclosure timestamps.",
            "Document manual source assumptions.",
            "Do not use in model training until reproducible.",
        ],
    },
    "capital_flow_money": {
        "category": "capital_flow",
        "difficulty": "medium",
        "action": "Add main, large-order, and divergence flow features only after source definitions are documented.",
        "outputs": [
            "main_net_inflow",
            "large_order_inflow",
            "small_order_inflow",
            "institutional_net_buy",
            "retail_net_buy",
            "money_flow_divergence",
            "flow_persistence_5d",
        ],
        "checks": [
            "Document vendor calculation method.",
            "Use end-of-day data only after market close.",
            "Check whether flow features duplicate volume signals.",
        ],
    },
    "chip_cost_distribution": {
        "category": "future_advanced",
        "difficulty": "high",
        "action": "Delay until a reproducible and legally usable chip-cost source is available.",
        "outputs": [
            "cost_concentration",
            "winner_ratio",
            "average_holding_cost",
            "chip_lock_up",
            "pressure_level",
            "support_level",
        ],
        "checks": [
            "Verify source rights and formulas.",
            "Snapshot values at prediction time.",
            "Test sensitivity to opaque vendor assumptions.",
        ],
    },
    "industry_style_factors": {
        "category": "market_structure",
        "difficulty": "medium",
        "action": "Add industry, style, rotation, and benchmark relative strength features using historical classifications.",
        "outputs": [
            "sector_membership",
            "industry_momentum",
            "industry_rotation_score",
            "style_value_score",
            "style_quality_score",
            "benchmark_relative_strength",
        ],
        "checks": [
            "Avoid applying current industry membership backward.",
            "Align industry index dates.",
            "Compare stock return against industry return.",
        ],
    },
    "index_component_factors": {
        "category": "market_structure",
        "difficulty": "medium",
        "action": "Add index membership and rebalance features from announced constituent files.",
        "outputs": [
            "csi300_member",
            "csi500_member",
            "csi1000_member",
            "index_weight",
            "index_inclusion_event",
            "passive_flow_proxy",
        ],
        "checks": [
            "Use announced rebalance dates only.",
            "Do not use future index membership before announcement.",
            "Check membership history by symbol and date.",
        ],
    },
    "macro_policy": {
        "category": "macro_liquidity",
        "difficulty": "medium",
        "action": "Add broad macro and liquidity context after single-stock and industry features are stable.",
        "outputs": [
            "interest_rate",
            "exchange_rate",
            "money_market_liquidity",
            "credit_spread",
            "cpi",
            "ppi",
            "pmi",
            "risk_appetite_proxy",
        ],
        "checks": [
            "Use release dates, not economic period labels.",
            "Forward-fill only after publication date.",
            "Measure whether macro adds value beyond market regime features.",
        ],
    },
    "sentiment_news_event": {
        "category": "sentiment_news",
        "difficulty": "high",
        "action": "Keep as research-only until timestamped text sources and source rights are clear.",
        "outputs": [
            "news_count",
            "positive_sentiment",
            "negative_sentiment",
            "event_tags",
            "policy_news_flag",
            "earnings_announcement_news",
            "social_media_heat",
            "abnormal_attention",
        ],
        "checks": [
            "Timestamp every text item.",
            "Exclude news after prediction time.",
            "Separate event tags from labels derived from future returns.",
        ],
    },
    "market_regime_timing": {
        "category": "market_structure",
        "difficulty": "medium",
        "action": "Add index trend, breadth, limit-up/down, and risk-on/risk-off regime features using past market data.",
        "outputs": [
            "index_trend",
            "market_breadth",
            "up_down_ratio",
            "limit_up_count",
            "limit_down_count",
            "market_regime",
            "risk_on_off_signal",
        ],
        "checks": [
            "Calculate regime labels from past data only.",
            "Check breadth source completeness.",
            "Use walk-forward validation across regimes.",
        ],
    },
    "custom_ai_factors": {
        "category": "future_advanced",
        "difficulty": "high",
        "action": "Keep advanced composite, drift, online-learning, and reinforcement-learning signals out of baseline training until inputs are auditable.",
        "outputs": [
            "multi_factor_confidence_score",
            "model_drift_indicator",
            "self_training_feedback_signal",
            "online_learning_candidate",
            "reinforcement_learning_candidate",
        ],
        "checks": [
            "Version every component score.",
            "Do not feed future realized performance back into same-period features.",
            "Validate with walk-forward and out-of-symbol tests.",
        ],
    },
}

SUPPLEMENTAL_ITEMS = [
    {
        "category": "risk_trading_feasibility",
        "feature_group": "risk_trading_feasibility",
        "feature_name": "tradability_constraints",
        "source_name": "Daily trading status and OHLCV limits",
        "data_frequency": "daily/event",
        "priority": "P1",
        "training_value": "medium",
        "implementation_difficulty": "medium",
        "leakage_risk": "medium",
        "token_required": False,
        "cost_level": "free",
        "recommended_action": "Add suspension, ST, limit-up/down, liquidity, transaction-cost, slippage, and turnover-capacity flags before using ML signals in backtests.",
        "output_columns": [
            "suspension_risk",
            "st_risk",
            "limit_up_constraint",
            "limit_down_constraint",
            "liquidity_filter",
            "transaction_cost_sensitivity",
            "slippage_sensitivity",
            "turnover_capacity",
        ],
        "validation_checks": [
            "Use only same-day status available before execution.",
            "Confirm constraints are not derived from future trade outcomes.",
            "Compare model signal performance before and after tradability filters.",
        ],
        "notes": "This improves realistic research and backtest feasibility; it does not change current backtester behavior.",
    },
    {
        "category": "future_advanced",
        "feature_group": "advanced_research_methods",
        "feature_name": "alternative_and_adaptive_signals",
        "source_name": "Future research pipeline",
        "data_frequency": "future",
        "priority": "P3",
        "training_value": "experimental",
        "implementation_difficulty": "high",
        "leakage_risk": "high",
        "token_required": False,
        "cost_level": "unknown",
        "recommended_action": "Delay alternative data, multi-agent research signals, self-training feedback, online learning, and reinforcement-learning candidates until the baseline factor pipeline is stable.",
        "output_columns": [
            "alternative_data_signal",
            "multi_agent_research_signal",
            "self_training_feedback_signal",
            "online_learning_candidate",
            "reinforcement_learning_candidate",
        ],
        "validation_checks": [
            "Require reproducible source snapshots.",
            "Separate training feedback from future realized returns.",
            "Use strict walk-forward validation and out-of-sample symbol tests.",
        ],
        "notes": "Research-only placeholder. These features can easily overfit or leak future outcomes.",
    },
]


def normalize_training_value(value: str) -> str:
    if value == "unknown":
        return "experimental"
    if value in {"high", "medium", "low", "experimental"}:
        return value
    return "experimental"


def calculate_implementation_score(
    priority: str,
    training_value: str,
    leakage_risk: str,
    cost_level: str,
    token_required: bool,
    implementation_difficulty: str,
) -> int:
    return (
        PRIORITY_SCORE.get(priority, 0)
        + TRAINING_VALUE_SCORE.get(training_value, 0)
        + LEAKAGE_SCORE.get(leakage_risk, -30)
        + COST_SCORE.get(cost_level, -5)
        + TOKEN_SCORE[token_required]
        + DIFFICULTY_SCORE.get(implementation_difficulty, -10)
    )


def make_queue_item(queue_id: str, payload: dict[str, Any]) -> FeatureQueueItem:
    priority = payload["priority"]
    stage = PRIORITY_STAGE.get(priority, "P3_research_only")
    training_value = normalize_training_value(payload["training_value"])
    score = calculate_implementation_score(
        priority=priority,
        training_value=training_value,
        leakage_risk=payload["leakage_risk"],
        cost_level=payload["cost_level"],
        token_required=payload["token_required"],
        implementation_difficulty=payload["implementation_difficulty"],
    )
    return FeatureQueueItem(
        queue_id=queue_id,
        priority=priority,
        category=payload["category"],
        feature_group=payload["feature_group"],
        feature_name=payload["feature_name"],
        source_name=payload["source_name"],
        data_frequency=payload["data_frequency"],
        training_value=training_value,
        implementation_stage=stage,
        implementation_difficulty=payload["implementation_difficulty"],
        leakage_risk=payload["leakage_risk"],
        token_required=payload["token_required"],
        cost_level=payload["cost_level"],
        recommended_action=payload["recommended_action"],
        output_columns=payload["output_columns"],
        validation_checks=payload["validation_checks"],
        notes=payload["notes"],
        implementation_score=score,
    )


def build_feature_queue() -> list[dict[str, Any]]:
    queue_items = []
    for registry_item in get_all_features():
        family = registry_item["factor_family"]
        details = QUEUE_DETAILS.get(family)
        if details is None:
            continue
        payload = {
            "priority": registry_item["implementation_priority"],
            "category": details["category"],
            "feature_group": family,
            "feature_name": registry_item["factor_name"],
            "source_name": registry_item["source_name"],
            "data_frequency": registry_item["update_frequency"],
            "training_value": registry_item["expected_predictive_value"],
            "implementation_difficulty": details["difficulty"],
            "leakage_risk": registry_item["leakage_risk"],
            "token_required": registry_item["requires_token"],
            "cost_level": registry_item["cost_level"],
            "recommended_action": details["action"],
            "output_columns": details["outputs"],
            "validation_checks": details["checks"],
            "notes": registry_item["notes"],
        }
        queue_id = f"FQ{len(queue_items) + 1:03d}"
        queue_items.append(make_queue_item(queue_id, payload))

    for supplemental in SUPPLEMENTAL_ITEMS:
        queue_id = f"FQ{len(queue_items) + 1:03d}"
        queue_items.append(make_queue_item(queue_id, supplemental))

    rows = [asdict(item) for item in queue_items]
    return sorted(
        rows,
        key=lambda row: (
            -row["implementation_score"],
            LEAKAGE_SORT.get(row["leakage_risk"], 9),
            STAGE_SORT.get(row["implementation_stage"], 9),
            row["queue_id"],
        ),
    )


def _stringify_list(value: Any) -> Any:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return value


def queue_to_dataframe(queue: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    records = build_feature_queue() if queue is None else queue
    display_rows = []
    for record in records:
        row = record.copy()
        row["output_columns"] = _stringify_list(row["output_columns"])
        row["validation_checks"] = _stringify_list(row["validation_checks"])
        display_rows.append(row)
    return pd.DataFrame(display_rows)


def summarize_feature_queue(queue: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = build_feature_queue() if queue is None else queue
    return {
        "total_items": len(records),
        "p0_item_count": sum(row["implementation_stage"] == "P0_now" for row in records),
        "low_leakage_item_count": sum(row["leakage_risk"] == "low" for row in records),
        "token_free_item_count": sum(not row["token_required"] for row in records),
        "high_training_value_item_count": sum(row["training_value"] == "high" for row in records),
        "research_only_item_count": sum(row["implementation_stage"] == "P3_research_only" for row in records),
    }


def filter_feature_queue(
    queue: list[dict[str, Any]] | None = None,
    priority: str | None = None,
    category: str | None = None,
    leakage_risk: str | None = None,
    token_required: bool | None = None,
    implementation_difficulty: str | None = None,
) -> list[dict[str, Any]]:
    records = build_feature_queue() if queue is None else list(queue)
    if priority:
        records = [
            row
            for row in records
            if row["implementation_stage"] == priority or row["priority"] == priority
        ]
    if category:
        records = [row for row in records if row["category"] == category]
    if leakage_risk:
        records = [row for row in records if row["leakage_risk"] == leakage_risk]
    if token_required is not None:
        records = [row for row in records if row["token_required"] == token_required]
    if implementation_difficulty:
        records = [
            row
            for row in records
            if row["implementation_difficulty"] == implementation_difficulty
        ]
    return records


def export_feature_queue_csv(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    queue_to_dataframe().to_csv(path, index=False, encoding="utf-8-sig")
    return path
