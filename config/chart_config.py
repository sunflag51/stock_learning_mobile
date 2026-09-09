"""
Plotlyチャートの共通設定。

スマートフォンでの移動・拡大・縮小を妨げないように、
fixedrange=Falseとdragmode='pan'を基本設定にします。
"""


# =========================================================
# Plotly操作設定
# =========================================================

PLOTLY_CONFIG = {
    "responsive": True,
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "showTips": True,
    "modeBarButtonsToAdd": [
        "pan2d",
        "zoom2d",
        "zoomIn2d",
        "zoomOut2d",
        "resetScale2d",
    ],
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
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
# チャート共通レイアウト
# =========================================================

CHART_LAYOUT_DEFAULTS = {
    "template": "plotly_white",
    "dragmode": "pan",
    "hovermode": "x unified",
    "autosize": True,
    "margin": {
        "l": 8,
        "r": 8,
        "t": 45,
        "b": 30,
    },
    "legend": {
        "orientation": "h",
        "x": 0,
        "y": 1.02,
        "xanchor": "left",
        "yanchor": "bottom",
        "font": {
            "size": 10,
        },
    },
    "font": {
        "family": "Arial, sans-serif",
        "size": 11,
        "color": "#17202A",
    },
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
}


# =========================================================
# X軸
# =========================================================

XAXIS_DEFAULTS = {
    "fixedrange": False,
    "showgrid": True,
    "gridcolor": "#E8EDF3",
    "gridwidth": 1,
    "showspikes": True,
    "spikemode": "across",
    "spikesnap": "cursor",
    "spikedash": "dot",
    "spikecolor": "#607D8B",
    "spikethickness": 1,
    "rangeslider": {
        "visible": True,
        "thickness": 0.08,
    },
}


# =========================================================
# Y軸
# =========================================================

YAXIS_DEFAULTS = {
    "fixedrange": False,
    "autorange": True,
    "showgrid": True,
    "gridcolor": "#E8EDF3",
    "gridwidth": 1,
    "showspikes": True,
    "spikemode": "across",
    "spikesnap": "cursor",
    "spikedash": "dot",
    "spikecolor": "#607D8B",
    "spikethickness": 1,
    "side": "right",
}


# =========================================================
# 色
# =========================================================

CHART_COLORS = {
    "up": "#D32F2F",
    "down": "#1976D2",
    "sma_short": "#F57C00",
    "sma_middle": "#7B1FA2",
    "sma_long": "#212121",
    "bb_middle": "#455A64",
    "bb_upper": "#8E24AA",
    "bb_lower": "#8E24AA",
    "kc_middle": "#00897B",
    "kc_upper": "#43A047",
    "kc_lower": "#43A047",
    "volume_up": "rgba(211, 47, 47, 0.55)",
    "volume_down": "rgba(25, 118, 210, 0.55)",
    "rsi": "#5E35B1",
    "macd": "#1565C0",
    "macd_signal": "#EF6C00",
    "positive": "#2E7D32",
    "warning": "#F9A825",
    "negative": "#C62828",
    "neutral": "#78909C",
}


# =========================================================
# 表示期間ボタン
# =========================================================

RANGE_SELECTOR_BUTTONS = [
    {
        "count": 1,
        "label": "1か月",
        "step": "month",
        "stepmode": "backward",
    },
    {
        "count": 3,
        "label": "3か月",
        "step": "month",
        "stepmode": "backward",
    },
    {
        "count": 6,
        "label": "6か月",
        "step": "month",
        "stepmode": "backward",
    },
    {
        "count": 1,
        "label": "1年",
        "step": "year",
        "stepmode": "backward",
    },
    {
        "label": "全期間",
        "step": "all",
    },
]
