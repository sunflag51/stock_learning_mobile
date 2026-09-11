from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from plotly.subplots import make_subplots


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="株式学習アプリ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# スマートフォン表示の調整
# =========================================================
st.markdown(
    """
    <style>
    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        overscroll-behavior-y: contain;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-left: 0.7rem;
        padding-right: 0.7rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 1.75rem !important;
    }

    h2 {
        font-size: 1.35rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    div[data-testid="stPlotlyChart"] {
        width: 100%;
        overflow: hidden;
        border-radius: 10px;
    }

    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container,
    div[data-testid="stPlotlyChart"] .svg-container {
        width: 100% !important;
    }

    div[data-testid="stDataFrame"] {
        width: 100%;
    }

    .stButton button,
    .stDownloadButton button {
        width: 100%;
        min-height: 44px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.7rem;
            padding-left: 0.35rem;
            padding-right: 0.35rem;
        }

        h1 {
            font-size: 1.45rem !important;
        }

        h2 {
            font-size: 1.2rem !important;
        }

        h3 {
            font-size: 1.05rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 基本設定
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "assets" / "watchlist.csv"

REQUIRED_COLUMNS = [
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

TEXT_COLUMNS = [
    "enabled",
    "name",
    "symbol",
    "provider_symbol",
    "market",
    "category",
    "currency",
    "note",
]

TRUE_VALUES = {
    "true",
    "1",
    "yes",
    "y",
    "on",
    "有効",
}

FALSE_VALUES = {
    "false",
    "0",
    "no",
    "n",
    "off",
    "無効",
}

PLOTLY_CONFIG = {
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
    "displayModeBar": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "stock_chart",
        "height": 900,
        "width": 1400,
        "scale": 1,
    },
}


# =========================================================
# 補助関数
# =========================================================
def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\u3000", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")

    return text.strip()


def parse_enabled(value):
    normalized_value = clean_text(value).lower()

    if normalized_value in TRUE_VALUES:
        return True

    if normalized_value in FALSE_VALUES:
        return False

    return pd.NA


def format_number(value, decimals=2) -> str:
    if pd.isna(value):
        return "-"

    return f"{float(value):,.{decimals}f}"


def format_integer(value) -> str:
    if pd.isna(value):
        return "-"

    return f"{int(float(value)):,}"


# =========================================================
# CSV読込・検証
# =========================================================
@st.cache_data(show_spinner=False)
def load_watchlist(csv_path_string: str) -> pd.DataFrame:
    csv_path = Path(csv_path_string)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSVファイルが見つかりません：{csv_path}"
        )

    master_df = pd.read_csv(
        csv_path,
        dtype=str,
        encoding="utf-8-sig",
    )

    master_df.columns = [
        clean_text(column)
        for column in master_df.columns
    ]

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in master_df.columns
    ]

    if missing_columns:
        missing_text = "、".join(missing_columns)

        raise ValueError(
            "CSVに必要な列がありません："
            f"{missing_text}"
        )

    master_df = master_df[REQUIRED_COLUMNS].copy()

    for column in TEXT_COLUMNS:
        master_df[column] = master_df[column].map(clean_text)

    master_df["display_order"] = pd.to_numeric(
        master_df["display_order"],
        errors="coerce",
    )

    master_df["enabled_parsed"] = master_df["enabled"].map(
        parse_enabled
    )

    invalid_enabled_df = master_df.loc[
        master_df["enabled_parsed"].isna()
    ]

    if not invalid_enabled_df.empty:
        invalid_rows = (
            invalid_enabled_df.index
            .to_series()
            .add(2)
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "enabled列をtrueまたはfalseとして判定できません。"
            "確認するCSV行："
            + "、".join(invalid_rows)
        )

    empty_required_rows = master_df.loc[
        master_df[
            [
                "name",
                "symbol",
                "provider_symbol",
            ]
        ]
        .eq("")
        .any(axis=1)
    ]

    if not empty_required_rows.empty:
        invalid_rows = (
            empty_required_rows.index
            .to_series()
            .add(2)
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "name、symbol、provider_symbolのいずれかが空です。"
            "確認するCSV行："
            + "、".join(invalid_rows)
        )

    duplicated_symbols = master_df.loc[
        master_df["symbol"].duplicated(keep=False),
        "symbol",
    ].tolist()

    if duplicated_symbols:
        duplicated_text = "、".join(
            sorted(set(duplicated_symbols))
        )

        raise ValueError(
            "symbol列に重複があります："
            f"{duplicated_text}"
        )

    master_df["enabled"] = (
        master_df["enabled_parsed"]
        .astype(bool)
    )

    master_df = master_df.drop(
        columns=["enabled_parsed"]
    )

    master_df = master_df.sort_values(
        by=["display_order", "symbol"],
        na_position="last",
    ).reset_index(drop=True)

    return master_df


# =========================================================
# 株価データ取得
# =========================================================
@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def download_price_data(
    provider_symbol: str,
    period: str,
) -> pd.DataFrame:
    price_df = yf.download(
        tickers=provider_symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if price_df is None or price_df.empty:
        return pd.DataFrame()

    price_df = price_df.copy()

    if isinstance(price_df.columns, pd.MultiIndex):
        first_level = price_df.columns.get_level_values(0)
        second_level = price_df.columns.get_level_values(1)

        if provider_symbol in first_level:
            price_df = price_df.xs(
                provider_symbol,
                axis=1,
                level=0,
            )
        elif provider_symbol in second_level:
            price_df = price_df.xs(
                provider_symbol,
                axis=1,
                level=1,
            )
        else:
            price_df.columns = [
                clean_text(column[0])
                for column in price_df.columns
            ]

    price_df = price_df.reset_index()

    date_candidates = [
        "Date",
        "Datetime",
        "index",
    ]

    date_column = None

    for candidate in date_candidates:
        if candidate in price_df.columns:
            date_column = candidate
            break

    if date_column is None:
        date_column = price_df.columns[0]

    price_df = price_df.rename(
        columns={
            date_column: "Date",
        }
    )

    required_price_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_price_columns = [
        column
        for column in required_price_columns
        if column not in price_df.columns
    ]

    if missing_price_columns:
        return pd.DataFrame()

    price_df = price_df[
        required_price_columns
    ].copy()

    price_df["Date"] = pd.to_datetime(
        price_df["Date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

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

    price_df = price_df.dropna(
        subset=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    price_df = price_df.loc[
        (price_df["Open"] > 0)
        & (price_df["High"] > 0)
        & (price_df["Low"] > 0)
        & (price_df["Close"] > 0)
    ].copy()

    price_df["Volume"] = (
        price_df["Volume"]
        .fillna(0)
        .clip(lower=0)
    )

    price_df = price_df.drop_duplicates(
        subset=["Date"],
        keep="last",
    )

    price_df = price_df.sort_values(
        by="Date"
    ).reset_index(drop=True)

    return price_df


# =========================================================
# 指標計算
# =========================================================
def calculate_rsi(
    close_series: pd.Series,
    period: int = 14,
) -> pd.Series:
    price_change = close_series.diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100
        / (1 + relative_strength)
    )

    no_loss = (
        average_loss.eq(0)
        & average_gain.gt(0)
    )

    no_gain = (
        average_gain.eq(0)
        & average_loss.gt(0)
    )

    unchanged = (
        average_gain.eq(0)
        & average_loss.eq(0)
    )

    rsi = rsi.mask(no_loss, 100)
    rsi = rsi.mask(no_gain, 0)
    rsi = rsi.mask(unchanged, 50)

    return rsi


def add_indicators(
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    result_df = price_df.copy()

    result_df["SMA5"] = (
        result_df["Close"]
        .rolling(window=5)
        .mean()
    )

    result_df["SMA25"] = (
        result_df["Close"]
        .rolling(window=25)
        .mean()
    )

    result_df["RSI14"] = calculate_rsi(
        result_df["Close"],
        period=14,
    )

    result_df["Daily_Change"] = (
        result_df["Close"].diff()
    )

    result_df["Daily_Change_Pct"] = (
        result_df["Close"]
        .pct_change(fill_method=None)
        .mul(100)
    )

    return result_df


# =========================================================
# 全銘柄の取得
# =========================================================
def collect_stock_data(
    target_master: pd.DataFrame,
    period: str,
):
    stock_data = {}
    error_rows = []

    total_count = len(target_master)
    progress_bar = st.progress(0)
    status_area = st.empty()

    for position, row in target_master.iterrows():
        name = row["name"]
        symbol = row["symbol"]
        provider_symbol = row["provider_symbol"]

        status_area.info(
            f"取得中：{name}（{symbol}）"
        )

        try:
            price_df = download_price_data(
                provider_symbol=provider_symbol,
                period=period,
            )

            if price_df.empty:
                error_rows.append(
                    {
                        "銘柄": name,
                        "銘柄コード": symbol,
                        "取得コード": provider_symbol,
                        "内容": "価格データを取得できませんでした",
                    }
                )
            else:
                stock_data[symbol] = add_indicators(
                    price_df
                )

        except Exception as error:
            error_rows.append(
                {
                    "銘柄": name,
                    "銘柄コード": symbol,
                    "取得コード": provider_symbol,
                    "内容": str(error),
                }
            )

        progress_bar.progress(
            (position + 1) / total_count
        )

    status_area.empty()
    progress_bar.empty()

    error_df = pd.DataFrame(error_rows)

    return stock_data, error_df


# =========================================================
# 一覧表作成
# =========================================================
def create_summary_df(
    target_master: pd.DataFrame,
    stock_data: dict,
) -> pd.DataFrame:
    summary_rows = []

    for _, master_row in target_master.iterrows():
        symbol = master_row["symbol"]

        if symbol not in stock_data:
            continue

        price_df = stock_data[symbol]

        if price_df.empty:
            continue

        latest_row = price_df.iloc[-1]

        if len(price_df) >= 2:
            previous_close = price_df.iloc[-2]["Close"]
            change_value = (
                latest_row["Close"]
                - previous_close
            )

            if previous_close != 0:
                change_percent = (
                    change_value
                    / previous_close
                    * 100
                )
            else:
                change_percent = np.nan
        else:
            change_value = np.nan
            change_percent = np.nan

        first_close = price_df.iloc[0]["Close"]

        if first_close != 0:
            period_change_percent = (
                latest_row["Close"]
                / first_close
                - 1
            ) * 100
        else:
            period_change_percent = np.nan

        summary_rows.append(
            {
                "表示順": master_row["display_order"],
                "銘柄": master_row["name"],
                "銘柄コード": symbol,
                "市場": master_row["market"],
                "分類": master_row["category"],
                "通貨": master_row["currency"],
                "最新日": latest_row["Date"],
                "終値": latest_row["Close"],
                "前日差": change_value,
                "前日比（%）": change_percent,
                "期間騰落率（%）": period_change_percent,
                "RSI14": latest_row["RSI14"],
                "出来高": latest_row["Volume"],
                "取得件数": len(price_df),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        return summary_df

    summary_df = summary_df.sort_values(
        by=["表示順", "銘柄コード"],
        na_position="last",
    ).reset_index(drop=True)

    return summary_df


def create_display_summary(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    display_df = summary_df.copy()

    if display_df.empty:
        return display_df

    display_df["最新日"] = (
        pd.to_datetime(display_df["最新日"])
        .dt.strftime("%Y-%m-%d")
    )

    display_df["終値"] = display_df["終値"].map(
        lambda value: format_number(value, 2)
    )

    display_df["前日差"] = display_df["前日差"].map(
        lambda value: format_number(value, 2)
    )

    display_df["前日比（%）"] = display_df[
        "前日比（%）"
    ].map(
        lambda value: format_number(value, 2)
    )

    display_df["期間騰落率（%）"] = display_df[
        "期間騰落率（%）"
    ].map(
        lambda value: format_number(value, 2)
    )

    display_df["RSI14"] = display_df["RSI14"].map(
        lambda value: format_number(value, 2)
    )

    display_df["出来高"] = display_df["出来高"].map(
        format_integer
    )

    display_df["取得件数"] = display_df[
        "取得件数"
    ].map(format_integer)

    return display_df[
        [
            "銘柄",
            "銘柄コード",
            "市場",
            "分類",
            "通貨",
            "最新日",
            "終値",
            "前日差",
            "前日比（%）",
            "期間騰落率（%）",
            "RSI14",
            "出来高",
            "取得件数",
        ]
    ]


# =========================================================
# 個別銘柄チャート
# =========================================================
def create_stock_chart(
    price_df: pd.DataFrame,
    name: str,
    symbol: str,
    currency: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[
            0.58,
            0.20,
            0.22,
        ],
        subplot_titles=(
            "ローソク足・移動平均線",
            "出来高",
            "RSI（14日）",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=price_df["Date"],
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name="価格",
            increasing_line_color="#d62728",
            decreasing_line_color="#1f77b4",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["SMA5"],
            mode="lines",
            name="5日移動平均",
            line={
                "color": "#ff7f0e",
                "width": 1.8,
            },
            hovertemplate=(
                "%{x|%Y-%m-%d}
"
                "5日移動平均：%{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["SMA25"],
            mode="lines",
            name="25日移動平均",
            line={
                "color": "#9467bd",
                "width": 1.8,
            },
            hovertemplate=(
                "%{x|%Y-%m-%d}
"
                "25日移動平均：%{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    volume_colors = np.where(
        price_df["Close"] >= price_df["Open"],
        "rgba(214, 39, 40, 0.65)",
        "rgba(31, 119, 180, 0.65)",
    )

    fig.add_trace(
        go.Bar(
            x=price_df["Date"],
            y=price_df["Volume"],
            name="出来高",
            marker_color=volume_colors,
            hovertemplate=(
                "%{x|%Y-%m-%d}
"
                "出来高：%{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["RSI14"],
            mode="lines",
            name="RSI14",
            line={
                "color": "#2ca02c",
                "width": 2,
            },
            hovertemplate=(
                "%{x|%Y-%m-%d}
"
                "RSI14：%{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="rgba(214, 39, 40, 0.7)",
        row=3,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="rgba(31, 119, 180, 0.7)",
        row=3,
        col=1,
    )

    fig.update_xaxes(
        fixedrange=False,
        rangeslider_visible=False,
    )

    fig.update_yaxes(
        title_text=f"価格（{currency}）",
        fixedrange=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="出来高",
        fixedrange=False,
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        fixedrange=False,
        row=3,
        col=1,
    )

    fig.update_layout(
        title=f"{name}（{symbol}）",
        height=800,
        margin={
            "l": 15,
            "r": 15,
            "t": 90,
            "b": 30,
        },
        template="plotly_white",
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        modebar={
            "orientation": "h",
        },
        uirevision=f"stock_chart_{symbol}",
    )

    return fig


# =========================================================
# 相対推移比較チャート
# =========================================================
def create_relative_chart(
    stock_data: dict,
    summary_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    for _, summary_row in summary_df.iterrows():
        symbol = summary_row["銘柄コード"]
        name = summary_row["銘柄"]

        if symbol not in stock_data:
            continue

        price_df = stock_data[symbol].copy()

        price_df = price_df.loc[
            price_df["Close"].notna()
            & (price_df["Close"] > 0)
        ].copy()

        if price_df.empty:
            continue

        first_close = price_df["Close"].iloc[0]

        price_df["Relative_Value"] = (
            price_df["Close"]
            / first_close
            * 100
        )

        fig.add_trace(
            go.Scatter(
                x=price_df["Date"],
                y=price_df["Relative_Value"],
                mode="lines",
                name=f"{name}（{symbol}）",
                line={
                    "width": 2.2,
                },
                hovertemplate=(
                    "%{x|%Y-%m-%d}
"
                    "相対値：%{y:.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="rgba(80, 80, 80, 0.6)",
        annotation_text="開始時点＝100",
        annotation_position="bottom right",
    )

    fig.update_xaxes(
        title_text="日付",
        fixedrange=False,
        rangeslider_visible=False,
    )

    fig.update_yaxes(
        title_text="相対値（開始時点＝100）",
        fixedrange=False,
    )

    fig.update_layout(
        title=(
            "処理対象銘柄の相対推移"
            "（各銘柄の開始時点＝100）"
        ),
        height=560,
        margin={
            "l": 15,
            "r": 15,
            "t": 100,
            "b": 30,
        },
        template="plotly_white",
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        modebar={
            "orientation": "h",
        },
        uirevision="relative_comparison",
    )

    return fig


# =========================================================
# ダウンロードデータ作成
# =========================================================
def create_download_df(
    target_master: pd.DataFrame,
    stock_data: dict,
) -> pd.DataFrame:
    download_frames = []

    for _, master_row in target_master.iterrows():
        symbol = master_row["symbol"]

        if symbol not in stock_data:
            continue

        export_df = stock_data[symbol].copy()

        export_df.insert(
            0,
            "currency",
            master_row["currency"],
        )

        export_df.insert(
            0,
            "category",
            master_row["category"],
        )

        export_df.insert(
            0,
            "market",
            master_row["market"],
        )

        export_df.insert(
            0,
            "provider_symbol",
            master_row["provider_symbol"],
        )

        export_df.insert(
            0,
            "symbol",
            symbol,
        )

        export_df.insert(
            0,
            "name",
            master_row["name"],
        )

        download_frames.append(export_df)

    if not download_frames:
        return pd.DataFrame()

    download_df = pd.concat(
        download_frames,
        ignore_index=True,
    )

    return download_df


# =========================================================
# メイン処理
# =========================================================
def main():
    st.title("📈 株式学習アプリ")

    st.caption(
        "CSVで有効化した銘柄の価格、移動平均、"
        "出来高、RSI、相対推移を確認します。"
    )

    st.info(
        "グラフは横方向へのドラッグ、範囲選択、"
        "ダブルタップによる表示リセットなどを利用できます。"
        "端末やブラウザによって、2本指操作の反応は異なります。"
    )

    # -----------------------------------------------------
    # CSV読込
    # -----------------------------------------------------
    try:
        master_df = load_watchlist(
            str(CSV_PATH)
        )

    except Exception as error:
        st.error(
            "銘柄CSVの読込中にエラーが発生しました。"
        )

        st.code(
            str(error),
            language="text",
        )

        st.markdown(
            "CSVの保存場所は次のとおりです。"
        )

        st.code(
            "assets/watchlist.csv",
            language="text",
        )

        st.stop()

    target_master = master_df.loc[
        master_df["enabled"]
    ].copy()

    excluded_master = master_df.loc[
        ~master_df["enabled"]
    ].copy()

    st.success(
        "CSV読込完了："
        f"処理対象{len(target_master)}件、"
        f"処理対象外{len(excluded_master)}件です。"
    )

    if target_master.empty:
        st.warning(
            "enabled=trueの銘柄がありません。"
        )

        st.stop()

    # -----------------------------------------------------
    # 登録内容
    # -----------------------------------------------------
    with st.expander(
        "登録銘柄を確認",
        expanded=False,
    ):
        registration_df = master_df.copy()

        registration_df["処理区分"] = np.where(
            registration_df["enabled"],
            "処理対象",
            "処理対象外",
        )

        registration_df = registration_df.rename(
            columns={
                "display_order": "表示順",
                "name": "銘柄",
                "symbol": "銘柄コード",
                "provider_symbol": "取得コード",
                "market": "市場",
                "category": "分類",
                "currency": "通貨",
                "note": "備考",
            }
        )

        st.dataframe(
            registration_df[
                [
                    "表示順",
                    "処理区分",
                    "銘柄",
                    "銘柄コード",
                    "取得コード",
                    "市場",
                    "分類",
                    "通貨",
                    "備考",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 取得期間
    # -----------------------------------------------------
    period_options = {
        "3か月": "3mo",
        "6か月": "6mo",
        "1年": "1y",
        "2年": "2y",
        "5年": "5y",
    }

    selected_period_label = st.selectbox(
        "表示期間",
        options=list(period_options.keys()),
        index=2,
    )

    selected_period = period_options[
        selected_period_label
    ]

    if st.button(
        "最新データを再取得",
        type="primary",
    ):
        download_price_data.clear()
        st.rerun()

    # -----------------------------------------------------
    # データ取得
    # -----------------------------------------------------
    with st.spinner(
        "株価データを取得・検証しています。"
    ):
        stock_data, error_df = collect_stock_data(
            target_master=target_master,
            period=selected_period,
        )

    if not error_df.empty:
        st.warning(
            "取得できなかった銘柄があります。"
        )

        st.dataframe(
            error_df,
            use_container_width=True,
            hide_index=True,
        )

    if not stock_data:
        st.error(
            "表示可能な株価データがありません。"
        )

        st.stop()

    summary_df = create_summary_df(
        target_master=target_master,
        stock_data=stock_data,
    )

    if summary_df.empty:
        st.error(
            "一覧表を作成できませんでした。"
        )

        st.stop()

    # -----------------------------------------------------
    # 最新状況
    # -----------------------------------------------------
    st.header("1．最新状況")

    display_summary_df = create_display_summary(
        summary_df
    )

    st.dataframe(
        display_summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "価格や指標は取得元データに基づく参考表示です。"
        "最新の市場情報はmoomooでご確認ください。"
    )

    # -----------------------------------------------------
    # 相対推移
    # -----------------------------------------------------
    st.header("2．相対推移比較")

    st.caption(
        "それぞれの銘柄について、取得期間の最初の終値を"
        "100として計算しています。通貨が異なる銘柄でも、"
        "値動きの比率を同じ基準で確認できます。"
    )

    relative_fig = create_relative_chart(
        stock_data=stock_data,
        summary_df=summary_df,
    )

    st.plotly_chart(
        relative_fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key="relative_chart",
    )

    # -----------------------------------------------------
    # 個別チャート
    # -----------------------------------------------------
    st.header("3．個別銘柄チャート")

    available_symbols = summary_df[
        "銘柄コード"
    ].tolist()

    symbol_labels = {
        row["銘柄コード"]: (
            f'{row["銘柄"]}（{row["銘柄コード"]}）'
        )
        for _, row in summary_df.iterrows()
    }

    selected_symbol = st.selectbox(
        "表示する銘柄",
        options=available_symbols,
        format_func=lambda symbol: symbol_labels[
            symbol
        ],
    )

    selected_master_row = target_master.loc[
        target_master["symbol"]
        == selected_symbol
    ].iloc[0]

    selected_price_df = stock_data[
        selected_symbol
    ]

    stock_fig = create_stock_chart(
        price_df=selected_price_df,
        name=selected_master_row["name"],
        symbol=selected_symbol,
        currency=selected_master_row["currency"],
    )

    st.plotly_chart(
        stock_fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=f"stock_chart_{selected_symbol}",
    )

    latest_row = selected_price_df.iloc[-1]

    metric_columns = st.columns(3)

    metric_columns[0].metric(
        label="最新終値",
        value=(
            f'{format_number(latest_row["Close"], 2)} '
            f'{selected_master_row["currency"]}'
        ),
    )

    metric_columns[1].metric(
        label="RSI14",
        value=format_number(
            latest_row["RSI14"],
            2,
        ),
    )

    metric_columns[2].metric(
        label="出来高",
        value=format_integer(
            latest_row["Volume"]
        ),
    )

    with st.expander(
        "個別銘柄の数値データを表示",
        expanded=False,
    ):
        detail_df = selected_price_df.copy()

        detail_df["Date"] = (
            detail_df["Date"]
            .dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # CSV保存
    # -----------------------------------------------------
    st.header("4．CSV保存")

    download_df = create_download_df(
        target_master=target_master,
        stock_data=stock_data,
    )

    if not download_df.empty:
        csv_data = download_df.to_csv(
            index=False,
            encoding="utf-8-sig",
        ).encode("utf-8-sig")

        st.download_button(
            label="全銘柄データをCSVで保存",
            data=csv_data,
            file_name="stock_analysis_data.csv",
            mime="text/csv",
        )

    st.caption(
        "本アプリは株式学習用です。表示内容は参考情報であり、"
        "特定の取引を推奨するものではありません。"
    )


# =========================================================
# アプリ開始
# =========================================================
if __name__ == "__main__":
    main()
