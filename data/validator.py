"""
監視銘柄CSVと価格データの検証機能。

このファイルでは、入力データに問題がないかを確認し、
後続処理で分かりにくいエラーが発生することを防ぎます。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import (
    MAX_WATCHLIST_ROWS,
    WATCHLIST_REQUIRED_COLUMNS,
)


class DataValidationError(ValueError):
    """入力データの検証に失敗した場合に使用する例外。"""


# =========================================================
# 共通補助関数
# =========================================================

def _text(value: Any) -> str:
    """
    値を前後空白のない文字列へ変換する。

    NoneやNaNは空文字列として扱います。
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _parse_enabled(value: Any) -> bool:
    """
    CSVのenabled列をTrueまたはFalseへ変換する。

    使用可能な値:
    true, false, 1, 0, yes, no, on, off
    """
    normalized = _text(value).lower()

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "on",
        "有効",
        "表示",
    }

    false_values = {
        "false",
        "0",
        "no",
        "n",
        "off",
        "無効",
        "非表示",
    }

    if normalized in true_values:
        return True

    if normalized in false_values:
        return False

    raise DataValidationError(
        f"enabled列の値「{value}」を判定できません。"
        "trueまたはfalseを入力してください。"
    )


def _normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    列名のBOM、前後空白、大文字・小文字の違いを整理する。
    """
    df = dataframe.copy()

    normalized_columns = [
        str(column).replace("\ufeff", "").strip().lower()
        for column in df.columns
    ]

    if len(normalized_columns) != len(set(normalized_columns)):
        raise DataValidationError(
            "同じ名前の列が複数存在します。CSVの見出しを確認してください。"
        )

    df.columns = normalized_columns

    return df


# =========================================================
# 監視銘柄CSV検証
# =========================================================

def validate_watchlist(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    監視銘柄CSVを検証し、アプリで使用できる形式へ整える。

    Parameters
    ----------
    dataframe:
        読み込んだ監視銘柄DataFrame。

    Returns
    -------
    pandas.DataFrame
        検証・整理済み監視銘柄DataFrame。

    Raises
    ------
    DataValidationError
        空データ、必須列不足、不正値、重複などがある場合。
    """
    if dataframe is None:
        raise DataValidationError(
            "監視銘柄CSVが読み込まれていません。"
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(
            "監視銘柄データがDataFrame形式ではありません。"
        )

    if dataframe.empty:
        raise DataValidationError(
            "監視銘柄CSVにデータがありません。"
        )

    df = _normalize_column_names(dataframe)

    missing_columns = [
        column
        for column in WATCHLIST_REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise DataValidationError(
            "監視銘柄CSVに必須列が不足しています。\n"
            f"不足列: {', '.join(missing_columns)}"
        )

    # 必須列だけを既定順序で取り出します。
    df = df[WATCHLIST_REQUIRED_COLUMNS].copy()

    # 完全な空行を削除します。
    text_check = df.apply(
        lambda column: column.map(_text)
    )

    non_empty_rows = text_check.apply(
        lambda row: any(value != "" for value in row),
        axis=1,
    )

    df = df.loc[non_empty_rows].copy()

    if df.empty:
        raise DataValidationError(
            "監視銘柄CSVには空行しかありません。"
        )

    if len(df) > MAX_WATCHLIST_ROWS:
        raise DataValidationError(
            "監視銘柄CSVの行数が上限を超えています。"
            f"上限は{MAX_WATCHLIST_ROWS}銘柄です。"
        )

    # 文字列列を整理します。
    text_columns = [
        "name",
        "symbol",
        "provider_symbol",
        "market",
        "category",
        "currency",
        "note",
    ]

    for column in text_columns:
        df[column] = df[column].map(_text)

    # 表示順を数値へ変換します。
    display_order = pd.to_numeric(
        df["display_order"],
        errors="coerce",
    )

    invalid_order_rows = df.index[display_order.isna()].tolist()

    if invalid_order_rows:
        csv_rows = [row_number + 2 for row_number in invalid_order_rows]

        raise DataValidationError(
            "display_order列に数値ではない値があります。"
            f"CSV行: {csv_rows}"
        )

    if (display_order < 1).any():
        raise DataValidationError(
            "display_order列には1以上の整数を入力してください。"
        )

    if ((display_order % 1) != 0).any():
        raise DataValidationError(
            "display_order列には小数ではなく整数を入力してください。"
        )

    df["display_order"] = display_order.astype(int)

    # enabled列を真偽値へ変換します。
    parsed_enabled: list[bool] = []

    for index, value in df["enabled"].items():
        try:
            parsed_enabled.append(_parse_enabled(value))
        except DataValidationError as error:
            raise DataValidationError(
                f"CSVの{index + 2}行目: {error}"
            ) from error

    df["enabled"] = parsed_enabled

    # 重要項目の空欄を確認します。
    required_text_columns = [
        "name",
        "symbol",
        "provider_symbol",
        "market",
        "currency",
    ]

    empty_messages: list[str] = []

    for column in required_text_columns:
        empty_indexes = df.index[df[column] == ""].tolist()

        if empty_indexes:
            csv_rows = [row_number + 2 for row_number in empty_indexes]
            empty_messages.append(
                f"{column}: CSV行{csv_rows}"
            )

    if empty_messages:
        raise DataValidationError(
            "監視銘柄CSVに入力されていない必須項目があります。\n"
            + "\n".join(empty_messages)
        )

    # enabled=trueの銘柄が一つ以上あるか確認します。
    if not df["enabled"].any():
        raise DataValidationError(
            "enabled=trueの監視銘柄がありません。"
            "少なくとも1銘柄を有効にしてください。"
        )

    # 表示用銘柄コードの重複を確認します。
    symbol_upper = df["symbol"].str.upper()

    duplicate_symbol_mask = symbol_upper.duplicated(
        keep=False
    )

    if duplicate_symbol_mask.any():
        duplicates = sorted(
            df.loc[duplicate_symbol_mask, "symbol"].unique().tolist()
        )

        raise DataValidationError(
            "symbol列に重複があります。\n"
            f"重複コード: {', '.join(duplicates)}"
        )

    # 有効銘柄について、取得用コードの重複を確認します。
    enabled_df = df.loc[df["enabled"]].copy()

    provider_upper = enabled_df["provider_symbol"].str.upper()

    duplicate_provider_mask = provider_upper.duplicated(
        keep=False
    )

    if duplicate_provider_mask.any():
        duplicates = sorted(
            enabled_df.loc[
                duplicate_provider_mask,
                "provider_symbol",
            ].unique().tolist()
        )

        raise DataValidationError(
            "有効銘柄のprovider_symbol列に重複があります。\n"
            f"重複コード: {', '.join(duplicates)}"
        )

    # 大文字・小文字を整理します。
    df["symbol"] = df["symbol"].str.upper()
    df["provider_symbol"] = df["provider_symbol"].str.upper()
    df["currency"] = df["currency"].str.upper()

    # 表示順で並べ替えます。
    df = df.sort_values(
        by=["display_order", "name"],
        kind="stable",
    ).reset_index(drop=True)

    return df


# =========================================================
# 価格データ検証
# =========================================================

def validate_price_data(
    dataframe: pd.DataFrame,
    symbol: str = "",
    minimum_rows: int = 2,
) -> pd.DataFrame:
    """
    Yahoo Financeから取得したOHLCVデータを検証・整理する。

    Parameters
    ----------
    dataframe:
        価格データ。
    symbol:
        エラーメッセージに表示する取得用銘柄コード。
    minimum_rows:
        最低限必要な価格データ行数。

    Returns
    -------
    pandas.DataFrame
        整理済みOHLCVデータ。
    """
    symbol_label = symbol or "指定銘柄"

    if dataframe is None:
        raise DataValidationError(
            f"{symbol_label}の価格データが返されませんでした。"
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(
            f"{symbol_label}の価格データ形式が正しくありません。"
        )

    if dataframe.empty:
        raise DataValidationError(
            f"{symbol_label}の価格データが空です。"
            "銘柄コード、期間、上場状況を確認してください。"
        )

    df = dataframe.copy()

    # 列名の前後空白を除去します。
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    standard_name_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "volume": "Volume",
    }

    rename_map = {}

    for column in df.columns:
        normalized = str(column).strip().lower()

        if normalized in standard_name_map:
            rename_map[column] = standard_name_map[normalized]

    df = df.rename(columns=rename_map)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise DataValidationError(
            f"{symbol_label}の価格データに必要な列がありません。\n"
            f"不足列: {', '.join(missing_columns)}"
        )

    output_columns = required_columns.copy()

    if "Adj Close" in df.columns:
        output_columns.append("Adj Close")

    df = df[output_columns].copy()

    # 日付インデックスを整えます。
    converted_dates = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    valid_date_mask = ~pd.isna(converted_dates)

    df = df.loc[valid_date_mask].copy()
    converted_dates = converted_dates[valid_date_mask]

    if df.empty:
        raise DataValidationError(
            f"{symbol_label}の価格データに有効な日付がありません。"
        )

    date_index = pd.DatetimeIndex(converted_dates)

    if date_index.tz is not None:
        date_index = date_index.tz_localize(None)

    df.index = date_index
    df.index.name = "Date"

    # 数値列を数値型へ変換します。
    for column in output_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # OHLCが欠けている行は分析に使えないため除外します。
    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    # 出来高欠損は0として保持します。
    df["Volume"] = df["Volume"].fillna(0)

    # 日付を昇順にし、同一日付は最後のデータを残します。
    df = (
        df.sort_index()
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
    )

    if df.empty:
        raise DataValidationError(
            f"{symbol_label}の有効な価格データがありません。"
        )

    # 価格が0以下の行は異常値として除外します。
    positive_price_mask = (
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
    )

    df = df.loc[positive_price_mask].copy()

    if df.empty:
        raise DataValidationError(
            f"{symbol_label}の価格データがすべて不正値です。"
        )

    # 高値と安値の関係を検証します。
    invalid_ohlc_mask = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    )

    if invalid_ohlc_mask.any():
        invalid_dates = [
            timestamp.strftime("%Y-%m-%d")
            for timestamp in df.index[invalid_ohlc_mask][:5]
        ]

        raise DataValidationError(
            f"{symbol_label}の価格データにOHLCの矛盾があります。\n"
            f"該当日例: {', '.join(invalid_dates)}"
        )

    # 出来高の負数を防ぎます。
    if (df["Volume"] < 0).any():
        raise DataValidationError(
            f"{symbol_label}の価格データに負の出来高があります。"
        )

    if len(df) < minimum_rows:
        raise DataValidationError(
            f"{symbol_label}の価格データが不足しています。"
            f"取得行数は{len(df)}行、最低必要行数は"
            f"{minimum_rows}行です。"
        )

    return df
