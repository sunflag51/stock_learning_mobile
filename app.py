from data.watchlist_loader import (
    get_enabled_watchlist,
    load_watchlist_from_repository,
)

watchlist = load_watchlist_from_repository()
enabled_watchlist = get_enabled_watchlist(watchlist)

print(enabled_watchlist)
