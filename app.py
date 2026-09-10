from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf


# =========================================================
# 画面設定
# =========================================================
st.set_page_config(
    page_title="株価データの検証・正規化",
    layout="wide",
)

st.title("第9段階：取得データの検証・正規化")


# =========================================================
# 文字列を正規化する関数
# =========================================================
def normalize_text(value):
    """
    前後の半角空白・全角空白・改行・タブなどを除去します。
    """
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\u3000", " ")
        .replace("\u00a0", " ")
        .strip()
    )


# =========================================================
# enabled列をTrue / Falseに変換する関数
# =========================================================
def normalize_enabled(value):
    """
    true、1、yes、on、enabledをTrueとして扱います。
    """
    normalized_value = normalize_text(value).lower()

    return normalized_value in {
        "true",
        "1",
        "yes",
        "y",
        "on",
        "enabled",
    }


# =========================================================
# CSVを読み込む関数
# =========================================================
@st.cache_data
def load_watchlist(csv_path_string):
    csv_path = Path(csv_path_string)

    master_df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype=str,
    )

    master_df.columns = [
        normalize_text(column)
        for column in master_df.columns
    ]

    required_columns = [
        "display_order",
        "enabled",
        "name",
        "symbol",
        "provider_symbol",
        "market",
        "category",
        "currency",
        "note",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in master_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSVに必要な列がありません："
            + ", ".join(missing_columns)
        )

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
        master_df[column] = master_df[column].map(normalize_text)

    master_df["enabled"] = master_df["enabled"].map(
        normalize_enabled
    )

    master_df["display_order"] = pd.to_numeric(
        master_df["display_order"],
        errors="coerce",
    )

    master_df = master_df.sort_values(
        by="display_order",
        na_position="last",
    ).reset_index(drop=True)

    return master_df


# =========================================================
# 株価データを取得する関数
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(provider_symbol):
    """
    provider_symbolを使用して日足データを取得します。
    """
    ticker = yf.Ticker(provider_symbol)

    price_df = ticker.history(
        period="6mo",
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    return price_df


# =========================================================
# 取得データを検証・正規化する関数
# =========================================================
def validate_and_normalize_price_data(
    raw_df,
    display_symbol,
    provider_symbol,
):
    result = {
        "status": "正常",
        "original_rows": 0,
        "normalized_rows": 0,
        "duplicate_rows": 0,
        "missing_rows": 0,
        "invalid_rows": 0,
        "message": "",
    }

    if raw_df is None or raw_df.empty:
        result["status"] = "取得データなし"
        result["message"] = "価格データを取得できませんでした。"
        return pd.DataFrame(), result

    result["original_rows"] = len(raw_df)

    price_df = raw_df.copy()

    # インデックスの日付を通常の列へ変換
    price_df = price_df.reset_index()

    # 最初の列がDateまたはDatetimeでない場合にも対応
    first_column = price_df.columns[0]

    if first_column not in {"Date", "Datetime"}:
        price_df = price_df.rename(
            columns={first_column: "Date"}
        )
    elif first_column == "Datetime":
        price_df = price_df.rename(
            columns={"Datetime": "Date"}
        )

    required_price_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_price_columns
        if column not in price_df.columns
    ]

    if missing_columns:
        result["status"] = "列不足"
        result["message"] = (
            "取得データに必要な列がありません："
            + ", ".join(missing_columns)
        )
        return pd.DataFrame(), result

    # 必要な列だけを残す
    price_df = price_df[required_price_columns].copy()

    # 日付を統一
    price_df["Date"] = pd.to_datetime(
        price_df["Date"],
        errors="coerce",
        utc=True,
    )

    price_df["Date"] = (
        price_df["Date"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    # 価格と出来高を数値へ変換
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        price_df[column] = pd.to_numeric(
            price_df[column],
            errors="coerce",
        )

    # 日付の重複を確認
    duplicate_mask = price_df.duplicated(
        subset=["Date"],
        keep="last",
    )

    result["duplicate_rows"] = int(duplicate_mask.sum())

    price_df = price_df.drop_duplicates(
        subset=["Date"],
        keep="last",
    )

    # 欠損行を確認
    missing_mask = price_df[
        required_price_columns
    ].isna().any(axis=1)

    result["missing_rows"] = int(missing_mask.sum())

    # OHLCと出来高の整合性を確認
    invalid_price_mask = (
        (price_df["Open"] <= 0)
        | (price_df["High"] <= 0)
        | (price_df["Low"] <= 0)
        | (price_df["Close"] <= 0)
        | (price_df["Volume"] < 0)
        | (
            price_df["High"]
            < price_df[
                ["Open", "Low", "Close"]
            ].max(axis=1)
        )
        | (
            price_df["Low"]
            > price_df[
                ["Open", "High", "Close"]
            ].min(axis=1)
        )
    )

    result["invalid_rows"] = int(
        invalid_price_mask.sum()
    )

    # 欠損行・不整合行を除外
    valid_mask = ~missing_mask & ~invalid_price_mask

    price_df = price_df.loc[valid_mask].copy()

    # 日付順に並べ替え
    price_df = price_df.sort_values(
        by="Date",
        ascending=True,
    ).reset_index(drop=True)

    # 銘柄情報を追加
    price_df.insert(
        0,
        "symbol",
        display_symbol,
    )

    price_df.insert(
        1,
        "provider_symbol",
        provider_symbol,
    )

    result["normalized_rows"] = len(price_df)

    if price_df.empty:
        result["status"] = "有効データなし"
        result["message"] = (
            "検証後に使用可能な価格データが残りませんでした。"
        )
    elif (
        result["duplicate_rows"] > 0
        or result["missing_rows"] > 0
        or result["invalid_rows"] > 0
    ):
        result["status"] = "修正済み"
        result["message"] = (
            "重複・欠損・不整合のある行を除外しました。"
        )
    else:
        result["status"] = "正常"
        result["message"] = (
            "欠損・重複・価格不整合は検出されませんでした。"
        )

    return price_df, result


# =========================================================
# メイン処理
# =========================================================
try:
    app_dir = Path(__file__).resolve().parent
    csv_path = app_dir / "assets" / "watchlist.csv"

    if not csv_path.exists():
        st.error(
            "CSVファイルが見つかりません。\n\n"
            f"確認した場所：`{csv_path}`"
        )
        st.stop()

    master_df = load_watchlist(str(csv_path))

    target_master = master_df.loc[
        master_df["enabled"] == True
    ].copy()

    excluded_master = master_df.loc[
        master_df["enabled"] == False
    ].copy()

    st.success(
        f"確認完了：処理対象{len(target_master)}件、"
        f"処理対象外{len(excluded_master)}件です。"
    )

    # -----------------------------------------------------
    # 対象銘柄を表示
    # -----------------------------------------------------
    st.subheader("1．処理対象銘柄")

    st.dataframe(
        target_master[
            [
                "display_order",
                "name",
                "symbol",
                "provider_symbol",
                "market",
                "currency",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # 株価データの取得・検証
    # -----------------------------------------------------
    st.subheader("2．価格データの取得・検証")

    all_normalized_data = []
    validation_results = []

    progress_bar = st.progress(0)
    status_area = st.empty()

    target_count = len(target_master)

    for position, (_, stock) in enumerate(
        target_master.iterrows(),
        start=1,
    ):
        name = stock["name"]
        display_symbol = stock["symbol"]
        provider_symbol = stock["provider_symbol"]

        status_area.info(
            f"取得中：{name}（{display_symbol}）"
        )

        try:
            raw_price_df = fetch_price_data(
                provider_symbol
            )

            normalized_df, result = (
                validate_and_normalize_price_data(
                    raw_df=raw_price_df,
                    display_symbol=display_symbol,
                    provider_symbol=provider_symbol,
                )
            )

        except Exception as error:
            normalized_df = pd.DataFrame()

            result = {
                "status": "取得エラー",
                "original_rows": 0,
                "normalized_rows": 0,
                "duplicate_rows": 0,
                "missing_rows": 0,
                "invalid_rows": 0,
                "message": str(error),
            }

        result["name"] = name
        result["symbol"] = display_symbol
        result["provider_symbol"] = provider_symbol

        validation_results.append(result)

        if not normalized_df.empty:
            all_normalized_data.append(normalized_df)

        progress_bar.progress(
            position / target_count
        )

    status_area.empty()

    # -----------------------------------------------------
    # 検証結果を表示
    # -----------------------------------------------------
    validation_df = pd.DataFrame(
        validation_results
    )

    validation_columns = [
        "name",
        "symbol",
        "provider_symbol",
        "status",
        "original_rows",
        "normalized_rows",
        "duplicate_rows",
        "missing_rows",
        "invalid_rows",
        "message",
    ]

    st.dataframe(
        validation_df[validation_columns],
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # 正規化後データを結合
    # -----------------------------------------------------
    st.subheader("3．正規化後のデータ")

    if all_normalized_data:
        normalized_all_df = pd.concat(
            all_normalized_data,
            ignore_index=True,
        )

        st.success(
            "処理対象銘柄の取得データを"
            "検証・正規化しました。"
        )

        st.write(
            f"正規化後の合計件数："
            f"{len(normalized_all_df):,}行"
        )

        # 各銘柄の最新5行を表示
        preview_df = (
            normalized_all_df
            .sort_values(
                ["symbol", "Date"],
                ascending=[True, False],
            )
            .groupby(
                "symbol",
                group_keys=False,
            )
            .head(5)
            .reset_index(drop=True)
        )

        st.caption(
            "以下は各銘柄の新しい日付から5行です。"
        )

        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
        )

        # CSVダウンロード
        csv_data = normalized_all_df.to_csv(
            index=False,
        ).encode("utf-8-sig")

        st.download_button(
            label="正規化後データをCSVで保存",
            data=csv_data,
            file_name="normalized_price_data.csv",
            mime="text/csv",
        )

    else:
        st.error(
            "正規化後の価格データがありません。"
            "検証結果のmessage列を確認してください。"
        )

    # -----------------------------------------------------
    # 対象外銘柄を表示
    # -----------------------------------------------------
    with st.expander("処理対象外の銘柄を確認"):
        st.dataframe(
            excluded_master[
                [
                    "display_order",
                    "name",
                    "symbol",
                    "provider_symbol",
                    "note",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

except Exception as error:
    st.error("処理中にエラーが発生しました。")
    st.exception(error)
