from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# =========================================================
# 1. Streamlit基本設定
# =========================================================
st.set_page_config(
    page_title="株価学習チャート",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. スマートフォン向け表示調整
# =========================================================
st.markdown(
    """
    <style>
    /* 画面左右の余白を小さくする */
    .block-container {
        padding-top: 1rem;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }

    /* Plotly上のタッチ操作を有効にする */
    .js-plotly-plot,
    .plotly,
    .plot-container {
        touch-action: none !important;
    }

    /* 操作ボタンを押しやすくする */
    .modebar-btn {
        transform: scale(1.15);
        margin-left: 3px !important;
        margin-right: 3px !important;
    }

    /* スマートフォン表示 */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.15rem;
            padding-right: 0.15rem;
        }

        h1 {
            font-size: 1.55rem !important;
        }

        h2,
        h3 {
            font-size: 1.15rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. 定数
# =========================================================
CSV_PATH = Path("assets/watchlist.csv")

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


# =========================================================
# 4. CSVを読み込む関数
# =========================================================
@st.cache_data
def load_watchlist(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 列名の前後空白を削除
    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSVに必要な列がありません："
            + ", ".join(missing_columns)
        )

    # 文字列列の前後空白を削除
    text_columns = [
        "enabled",
        "name",
        "symbol",
        "provider_symbol",
        "market",
        "category",
        "currency",
        "note",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    # enabledをTrue／Falseへ統一
    enabled_text = df["enabled"].str.lower()

    true_values = {"true", "1", "yes", "y", "on"}
    false_values = {"false", "0", "no", "n", "off"}

    unknown_values = set(enabled_text.unique()) - true_values - false_values

    if unknown_values:
        raise ValueError(
            "enabled列に判定できない値があります："
            + ", ".join(sorted(unknown_values))
        )

    df["enabled"] = enabled_text.isin(true_values)

    # 表示順を数値へ変換
    df["display_order"] = pd.to_numeric(
        df["display_order"],
        errors="coerce",
    )

    if df["display_order"].isna().any():
        raise ValueError(
            "display_order列に数値へ変換できない値があります。"
        )

    df = df.sort_values("display_order").reset_index(drop=True)

    return df


# =========================================================
# 5. 株価データを取得する関数
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def download_price_data(
    provider_symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    data = yf.download(
        tickers=provider_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # yfinanceでMultiIndexになった場合の処理
    if isinstance(data.columns, pd.MultiIndex):
        first_level = data.columns.get_level_values(0)

        if "Close" in first_level:
            data.columns = first_level
        else:
            data.columns = data.columns.get_level_values(-1)

    data = data.reset_index()

    # 日付列名をDateへ統一
    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})
    elif "index" in data.columns:
        data = data.rename(columns={"index": "Date"})

    required_price_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column for column in required_price_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "取得データに必要な列がありません："
            + ", ".join(missing_columns)
        )

    data = data[required_price_columns].copy()

    # 日付を正規化
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
        utc=True,
    )

    # タイムゾーン情報を除去
    data["Date"] = data["Date"].dt.tz_convert(None)

    # 数値列を正規化
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # 必須値が欠けている行を除外
    data = data.dropna(
        subset=["Date", "Open", "High", "Low", "Close"]
    )

    # 重複した日時を除外
    data = data.drop_duplicates(
        subset=["Date"],
        keep="last",
    )

    # 日付順へ並べ替え
    data = data.sort_values("Date").reset_index(drop=True)

    # OHLC整合性を確認
    valid_ohlc = (
        (data["High"] >= data[["Open", "Close", "Low"]].max(axis=1))
        & (data["Low"] <= data[["Open", "Close", "High"]].min(axis=1))
        & (data["Volume"].fillna(0) >= 0)
    )

    data = data.loc[valid_ohlc].copy()
    data["Volume"] = data["Volume"].fillna(0)

    return data.reset_index(drop=True)


# =========================================================
# 6. テクニカル指標を計算する関数
# =========================================================
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    # 移動平均線
    result["SMA5"] = result["Close"].rolling(
        window=5,
        min_periods=5,
    ).mean()

    result["SMA20"] = result["Close"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    result["SMA60"] = result["Close"].rolling(
        window=60,
        min_periods=60,
    ).mean()

    # RSI（14期間、Wilder方式）
    difference = result["Close"].diff()

    gain = difference.clip(lower=0)
    loss = -difference.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    result["RSI14"] = 100 - (
        100 / (1 + relative_strength)
    )

    # 下落がない期間はRSIを100として扱う
    result.loc[
        (average_loss == 0) & (average_gain > 0),
        "RSI14",
    ] = 100

    # 値動きがない期間はRSIを50として扱う
    result.loc[
        (average_loss == 0) & (average_gain == 0),
        "RSI14",
    ] = 50

    return result


# =========================================================
# 7. Plotlyチャートを作る関数
# =========================================================
def create_chart(
    data: pd.DataFrame,
    stock_name: str,
    symbol: str,
    currency: str,
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=(
            "ローソク足・移動平均線",
            "出来高",
            "RSI（14期間）",
        ),
    )

    # ローソク足
    figure.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="株価",
            increasing_line_color="#ef5350",
            increasing_fillcolor="#ef5350",
            decreasing_line_color="#26a69a",
            decreasing_fillcolor="#26a69a",
            whiskerwidth=0.4,
        ),
        row=1,
        col=1,
    )

    # 移動平均線
    figure.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["SMA5"],
            mode="lines",
            name="5日移動平均",
            line=dict(
                color="#ff9800",
                width=1.5,
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["SMA20"],
            mode="lines",
            name="20日移動平均",
            line=dict(
                color="#2196f3",
                width=1.7,
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["SMA60"],
            mode="lines",
            name="60日移動平均",
            line=dict(
                color="#9c27b0",
                width=1.7,
            ),
        ),
        row=1,
        col=1,
    )

    # 出来高の色
    volume_colors = np.where(
        data["Close"] >= data["Open"],
        "#ef5350",
        "#26a69a",
    )

    figure.add_trace(
        go.Bar(
            x=data["Date"],
            y=data["Volume"],
            name="出来高",
            marker_color=volume_colors,
            opacity=0.75,
        ),
        row=2,
        col=1,
    )

    # RSI
    figure.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["RSI14"],
            mode="lines",
            name="RSI 14",
            line=dict(
                color="#7e57c2",
                width=2,
            ),
        ),
        row=3,
        col=1,
    )

    # RSI基準線
    figure.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ef5350",
        line_width=1,
        row=3,
        col=1,
    )

    figure.add_hline(
        y=50,
        line_dash="dot",
        line_color="#9e9e9e",
        line_width=1,
        row=3,
        col=1,
    )

    figure.add_hline(
        y=30,
        line_dash="dash",
        line_color="#26a69a",
        line_width=1,
        row=3,
        col=1,
    )

    figure.update_layout(
        title=dict(
            text=f"{stock_name}（{symbol}）",
            x=0.01,
            xanchor="left",
            font=dict(size=18),
        ),
        height=820,
        autosize=True,
        template="plotly_white",

        # 1本指ではチャートを左右へ移動
        dragmode="pan",

        # 指を離したときの表示を軽くする
        hovermode="x unified",

        margin=dict(
            l=8,
            r=8,
            t=70,
            b=30,
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),

        # 下部の小さなレンジスライダーは非表示
        xaxis_rangeslider_visible=False,

        # 描画後の拡大位置を可能な範囲で維持
        uirevision=f"{symbol}-chart",
    )

    # すべての横軸で拡大・移動を許可
    figure.update_xaxes(
        fixedrange=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#777777",
    )

    # すべての縦軸で拡大・移動を許可
    figure.update_yaxes(
        fixedrange=False,
        showspikes=True,
        spikethickness=1,
        spikecolor="#777777",
    )

    figure.update_yaxes(
        title_text=f"価格（{currency}）",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="出来高",
        rangemode="tozero",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        row=3,
        col=1,
    )

    return figure


# =========================================================
# 8. メイン画面
# =========================================================
st.title("📈 株価学習チャート")

st.caption(
    "2本指：ピンチ拡大・縮小 ／ "
    "1本指：左右移動 ／ "
    "上部ツールバー：拡大・縮小・表示リセット"
)

# CSVの存在確認
if not CSV_PATH.exists():
    st.error(
        f"CSVファイルが見つかりません：{CSV_PATH}"
    )
    st.stop()

# CSV読込
try:
    master_df = load_watchlist(str(CSV_PATH))
except Exception as error:
    st.error("銘柄CSVの読み込みに失敗しました。")
    st.exception(error)
    st.stop()

# enabled=trueだけを処理対象にする
target_master = master_df.loc[
    master_df["enabled"] == True
].copy()

target_master = target_master.sort_values(
    "display_order"
).reset_index(drop=True)

if target_master.empty:
    st.warning(
        "enabled=trueの処理対象銘柄がありません。"
    )
    st.stop()

# 銘柄選択用表示名
target_master["selection_label"] = (
    target_master["name"]
    + "（"
    + target_master["symbol"]
    + "）"
)

selection_labels = target_master[
    "selection_label"
].tolist()

selected_label = st.selectbox(
    "表示する銘柄",
    options=selection_labels,
)

selected_row = target_master.loc[
    target_master["selection_label"] == selected_label
].iloc[0]

stock_name = selected_row["name"]
symbol = selected_row["symbol"]
provider_symbol = selected_row["provider_symbol"]
currency = selected_row["currency"]

# 表示期間
period_label = st.selectbox(
    "表示期間",
    options=[
        "3か月",
        "6か月",
        "1年",
        "2年",
        "5年",
    ],
    index=2,
)

period_map = {
    "3か月": "3mo",
    "6か月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "5年": "5y",
}

selected_period = period_map[period_label]

# 株価取得
try:
    with st.spinner(
        f"{stock_name}（{symbol}）のデータを取得しています..."
    ):
        price_data = download_price_data(
            provider_symbol=provider_symbol,
            period=selected_period,
            interval="1d",
        )
except Exception as error:
    st.error(
        f"{stock_name}（{symbol}）のデータ取得に失敗しました。"
    )
    st.exception(error)
    st.stop()

if price_data.empty:
    st.warning(
        f"{stock_name}（{symbol}）の株価データを取得できませんでした。"
    )
    st.stop()

# 指標計算
chart_data = calculate_indicators(price_data)

# 最新データ表示
latest = chart_data.iloc[-1]
previous = (
    chart_data.iloc[-2]
    if len(chart_data) >= 2
    else chart_data.iloc[-1]
)

price_change = latest["Close"] - previous["Close"]

if previous["Close"] != 0:
    price_change_rate = (
        price_change / previous["Close"]
    ) * 100
else:
    price_change_rate = np.nan

column1, column2, column3 = st.columns(3)

column1.metric(
    label=f"終値（{currency}）",
    value=f"{latest['Close']:,.2f}",
    delta=(
        f"{price_change:+,.2f} "
        f"({price_change_rate:+.2f}%)"
        if not np.isnan(price_change_rate)
        else None
    ),
)

column2.metric(
    label="出来高",
    value=f"{latest['Volume']:,.0f}",
)

column3.metric(
    label="RSI（14）",
    value=(
        f"{latest['RSI14']:.1f}"
        if pd.notna(latest["RSI14"])
        else "計算中"
    ),
)

# チャート作成
chart = create_chart(
    data=chart_data,
    stock_name=stock_name,
    symbol=symbol,
    currency=currency,
)

# Plotly操作設定
plotly_config = {
    # ピンチ・ホイール拡大を有効化
    "scrollZoom": True,

    # 画面幅に自動追従
    "responsive": True,

    # 操作ツールバーを常に表示
    "displayModeBar": True,

    # Plotlyロゴを非表示
    "displaylogo": False,

    # 操作ボタンを整理
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "toggleSpikelines",
    ],

    # 画像保存ボタンの設定
    "toImageButtonOptions": {
        "format": "png",
        "filename": f"{symbol}_chart",
        "height": 900,
        "width": 1400,
        "scale": 2,
    },

    # ダブルタップまたはダブルクリック時に初期表示へ戻す
    "doubleClick": "reset+autosize",
}

st.plotly_chart(
    chart,
    use_container_width=True,
    config=plotly_config,
)

with st.expander("チャートの操作方法"):
    st.markdown(
        """
        - **2本指を広げる**：ピンチアウトで拡大
        - **2本指を狭める**：ピンチインで縮小
        - **1本指で左右へ動かす**：表示期間を移動
        - **虫眼鏡ボタン**：範囲を指定して拡大
        - **手の形のボタン**：移動モード
        - **家の形のボタン**：初期表示へ戻す
        - **カメラボタン**：チャートを画像として保存
        """
    )

st.info(
    "株価データは外部の公開データ提供元から取得しており、"
    "リアルタイム価格とは限りません。最新の相場はmoomooでご確認ください。"
)

st.caption(
    "この画面は株価チャートと指標の学習用です。"
    "表示内容は参考情報であり、特定の取引を推奨するものではありません。"
)
