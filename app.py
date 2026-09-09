from data.market_provider import (
    fetch_price_data,
    get_price_data_summary,
)

price_data = fetch_price_data(
    provider_symbol="AAPL",
    period="2y",
    interval="1d",
)

print(price_data.tail())
print(get_price_data_summary(price_data))
