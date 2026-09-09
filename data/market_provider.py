"""
Yahoo Financeから価格データを取得する機能。

yfinanceを使用して、Open・High・Low・Close・Volumeを取得します。
取得後はvalidator.pyでデータを検証します。
"""

from __future__ import annotations

import re
import time
from typing import Final

import pandas as pd
import yfinance as yf

from config.settings import (
    DATA_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_DATA_INTERVAL,
    DEFAULT_DATA_PERIOD,
)
from data.validator import (
    DataValidationError,
    validate_price_data,
)


class MarketDataError(RuntimeError):
    """価格データの取得または検証に失敗した場合の例外。"""


ALLOWED_PERIODS: Final[set[str]] = {
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
}

ALLOWED_INTERVALS: Final[set[str]] = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}

# Yahoo Financeで一般的に使用される文字を許可します。
SYMBOL_PATTERN = re.compile(
    r"^[A-Za-z0-9^=._-]{1,40}$"
)


# =========================================================
# 入力値検証
# =========================================================

def validate_provider_symbol(symbol: str) -> str:
    """
    Yahoo Finance取得用銘柄コードを検証する。
    """
    normalized = str(symbol).strip().upper()

    if not normalized:
        raise MarketDataError(
            "価格取得用の銘柄コードが入力されていません。"
        )

    if not SYMBOL_PATTERN.fullmatch(normalized):
        raise MarketDataError(
            "価格取得用銘柄コードに使用できない文字があります。\n"
            f"入力値: {symbol}"
        )

    return normalized


def validate_period_and_interval(
    period: str,
    interval: str,
) -> tuple[str, str]:
    """
    Yahoo Financeの期間と足種を検証する。
    """
    normalized_period = str(period).strip().lower()
    normalized_interval = str(interval).strip().lower()

    if normalized_period not in ALLOWED_PERIODS:
        raise MarketDataError(
            f"未対応の取得期間です: {period}"
        )

    if normalized_interval not in ALLOWED_INTERVALS:
        raise MarketDataError(
            f"未対応の時間足です: {interval}"
        )

    return normalized_period, normalized_interval


# =========================================================
# yfinance列整理
# =========================================================

def _flatten_yfinance_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    yfinanceが返すMultiIndex列を通常のOHLCV列へ変換する。

    yfinanceのバージョンによっては、1銘柄でも以下のような
    複数階層列が返されることがあります。

    Price       Open High Low Close Volume
    Ticker      AAPL AAPL AAPL AAPL AAPL
    """
    if dataframe is None or dataframe.empty:
        return dataframe

    df = dataframe.copy()

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    desired_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    normalized_data: dict[str, pd.Series] = {}

    for desired in desired_columns:
        for column in df.columns:
            parts = [
                str(part).strip()
                for part in column
                if str(part).strip()
            ]

            matching_part = next(
                (
                    part
                    for part in parts
                    if part.lower() == desired.lower()
                ),
                None,
            )

            if matching_part is not None:
                selected = df[column]

                if isinstance(selected, pd.DataFrame):
                    selected = selected.iloc[:, 0]

                normalized_data[desired] = selected
                break

    if not normalized_data:
        return df

    return pd.DataFrame(
        normalized_data,
        index=df.index,
    )


# =========================================================
# 価格データ取得
# =========================================================

def _download_once(
    symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    """
    yfinance.downloadを1回実行する。
    """
    return yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
        timeout=DATA_REQUEST_TIMEOUT_SECONDS,
    )


def fetch_price_data(
    provider_symbol: str,
    period: str = DEFAULT_DATA_PERIOD,
    interval: str = DEFAULT_DATA_INTERVAL,
    minimum_rows: int = 2,
    retry_count: int = 2,
) -> pd.DataFrame:
    """
    Yahoo Financeから価格データを取得する。

    Parameters
    ----------
    provider_symbol:
        Yahoo Finance取得用コード。
        例: AAPL、GOOG、7974.T

    period:
        取得期間。
        例: 6mo、1y、2y、5y、max

    interval:
        時間足。
        例: 1d、1wk、1mo

    minimum_rows:
        最低限必要な価格データ行数。

    retry_count:
        一時的な通信失敗時の最大試行回数。

    Returns
    -------
    pandas.DataFrame
        検証済みOHLCVデータ。
    """
    symbol = validate_provider_symbol(
        provider_symbol
    )

    normalized_period, normalized_interval = (
        validate_period_and_interval(
            period,
            interval,
        )
    )

    retry_count = max(1, min(int(retry_count), 3))

    last_error: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            raw_data = _download_once(
                symbol=symbol,
                period=normalized_period,
                interval=normalized_interval,
            )

            raw_data = _flatten_yfinance_columns(
                raw_data
            )

            validated_data = validate_price_data(
                dataframe=raw_data,
                symbol=symbol,
                minimum_rows=minimum_rows,
            )

            validated_data.attrs["provider_symbol"] = symbol
            validated_data.attrs["period"] = normalized_period
            validated_data.attrs["interval"] = normalized_interval
            validated_data.attrs["data_source"] = "Yahoo Finance"
            validated_data.attrs["last_data_date"] = (
                validated_data.index.max()
            )

            return validated_data

        except DataValidationError as error:
            last_error = error

        except Exception as error:
            last_error = error

        if attempt < retry_count:
            time.sleep(1.0)

    error_detail = (
        str(last_error)
        if last_error is not None
        else "原因を確認できませんでした。"
    )

    raise MarketDataError(
        f"{symbol}の価格データ取得に失敗しました。\n"
        f"詳細: {error_detail}\n"
        "銘柄コード、取得期間、時間足、通信状態を確認してください。"
    ) from last_error


# =========================================================
# データ状態確認
# =========================================================

def get_price_data_summary(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """
    価格データの確認用情報を辞書で返す。
    """
    validated = validate_price_data(
        dataframe,
        minimum_rows=1,
    )

    first_date = validated.index.min()
    last_date = validated.index.max()
    latest = validated.iloc[-1]

    return {
        "first_date": first_date,
        "last_date": last_date,
        "row_count": len(validated),
        "latest_open": float(latest["Open"]),
        "latest_high": float(latest["High"]),
        "latest_low": float(latest["Low"]),
        "latest_close": float(latest["Close"]),
        "latest_volume": float(latest["Volume"]),
    }


def has_enough_rows_for_analysis(
    dataframe: pd.DataFrame,
    required_rows: int,
) -> bool:
    """
    指標計算に必要な行数があるか確認する。

    例:
    200日移動平均線では、余裕を含めて220行程度を
    確認するために使用できます。
    """
    if dataframe is None:
        return False

    if not isinstance(dataframe, pd.DataFrame):
        return False

    return len(dataframe) >= int(required_rows)
