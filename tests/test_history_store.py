from history_store import HistoryItem, _safe_item, default_history


def test_history_items_have_stable_metadata_ids():
    item = HistoryItem(name="upload.csv", details="50 rows · CSV", type="csv")
    assert _safe_item(item)["id"] == _safe_item(item)["id"]
    assert _safe_item(item)["key"] is None


def test_default_history_preserves_sidebar_shape():
    items = default_history()
    assert len(items) == 3
    assert {"id", "name", "details", "type", "key"}.issubset(items[0])
    assert items[2]["key"] == "bank_customers"


def test_clear_history_persists_empty_state():
    from history_store import clear_history, list_history, save_history
    save_history({"name": "test.csv", "details": "10 rows", "type": "csv"})
    assert clear_history() is True
    items, _ = list_history()
    assert items == []
