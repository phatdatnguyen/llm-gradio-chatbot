"""Unit tests for message conversion, local readers, and history persistence.
No network access.
"""

import base64
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import history_store
from chatbot import messages
from chatbot import readers

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AARAAA//8DAgH/"
    "6h1DkQAAAABJRU5ErkJggg=="
)


# --------------------------------------------------------------------- messages

def test_prepare_chat_messages_wraps_images():
    history = [{"role": "user", "content": "what is this", "image_url": "data:image/png;base64,AAA"}]
    payload = messages.prepare_chat_messages(history)
    assert payload[0]["content"][0] == {"type": "text", "text": "what is this"}
    assert payload[0]["content"][1]["image_url"]["url"].startswith("data:image/png")


def test_local_image_only_message_omits_empty_text_part():
    history = [{"role": "user", "content": "", "image_url": "data:image/png;base64,AAA"}]
    assert messages.prepare_chat_messages(history) == [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]


def test_replace_history_content_collapses_blocks_and_shows_images():
    history = [
        {"role": "user",
         "content": messages.build_document_block("report.pdf", "lots of text\nmore text")},
        {"role": "user", "content": "<<<LINK_CONTENT>>>\nURL: http://x.test\npage body\n<<<END_LINK>>>"},
        {"role": "user", "content": "look", "image_url": "data:image/png;base64,AAA"},
    ]
    replaced = messages.replace_history_content(history)
    assert replaced[0]["content"] == "[File: report.pdf]"
    assert replaced[1]["content"] == "[URL: http://x.test]"
    assert replaced[2]["content"].endswith("![Image](data:image/png;base64,AAA)")
    assert all("image_url" not in message for message in replaced)


# --------------------------------------------------------------------- readers

def test_read_excel_covers_every_sheet(tmp_path):
    """This whole path used to fail because openpyxl was not installed."""
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"b": [3]}).to_excel(writer, sheet_name="Second", index=False)

    text = readers.read_document(str(path))
    assert "First" in text and "Second" in text and "3" in text


def test_read_text_file_handles_utf8_and_cp1252(tmp_path):
    utf8_path = tmp_path / "a.txt"
    utf8_path.write_text("xin chào", encoding="utf-8")
    assert "chào" in readers.read_document(str(utf8_path))

    cp1252_path = tmp_path / "b.txt"
    cp1252_path.write_bytes("caf\xe9".encode("cp1252"))
    assert "caf" in readers.read_document(str(cp1252_path))


def test_empty_document_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(readers.EmptyDocumentError):
        readers.read_document(str(path))


def test_html_is_converted_to_text(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<h1>Title</h1><p>Body text</p>", encoding="utf-8")
    text = readers.read_document(str(path))
    assert "Title" in text and "Body text" in text


# --------------------------------------------------------------- history store

@pytest.fixture
def history_dir(tmp_path, monkeypatch):
    directory = tmp_path / "history"
    directory.mkdir()
    monkeypatch.setattr(history_store, "HISTORY_DIR", str(directory))
    return directory


def test_path_traversal_is_blocked():
    for attempt in ("../../etc/passwd", "..\\..\\windows\\system32"):
        safe = history_store.sanitize_history_file_name(attempt)
        assert "/" not in safe and "\\" not in safe
        assert os.path.dirname(history_store.history_file_path(safe)) == history_store.HISTORY_DIR


def test_save_externalizes_images_and_load_restores_them(history_dir):
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    history = [
        {"role": "user", "content": "a dog", "image_url": inline},
        {"role": "assistant", "content": "here you go"},
    ]

    history_store.save_history(history, "cute dog")

    saved_path = history_dir / "cute dog.json"
    stored = json.loads(saved_path.read_text(encoding="utf8"))
    assert stored[0]["image_url"] == "assets/cute dog/0000.png"
    assert (history_dir / "assets" / "cute dog" / "0000.png").read_bytes() == PNG_1PX
    assert saved_path.stat().st_size < 500, "JSON should no longer carry the image"

    loaded, _ = history_store.load_history("cute dog")
    assert loaded[0]["image_url"] == inline
    assert loaded[1]["content"] == "here you go"


def test_multiple_embedded_images_in_one_message_round_trip(history_dir):
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    history = [{"role": "assistant", "content": f"![a]({inline}) and ![b]({inline})"}]

    history_store.save_history(history, "two")
    stored = json.loads((history_dir / "two.json").read_text(encoding="utf8"))
    assert "0000_0.png" in stored[0]["content"] and "0000_1.png" in stored[0]["content"]

    loaded, _ = history_store.load_history("two")
    assert loaded[0]["content"] == history[0]["content"]


def test_legacy_link_and_generated_image_history_remains_readable_without_rewrite(history_dir):
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    link = "<<<LINK_CONTENT>>>\nURL: https://example.test/article\nArchived page body\n<<<END_LINK>>>"
    legacy = [{"role": "user", "content": link, "image_url": inline},
              {"role": "assistant", "content": f"Archived image.\n\n![Image]({inline})"}]
    path = history_dir / "legacy mixed.json"
    saved_bytes = json.dumps(legacy).encode("utf-8")
    path.write_bytes(saved_bytes)
    loaded, _ = history_store.load_history("legacy mixed")
    assert loaded == legacy and path.read_bytes() == saved_bytes
    displayed = messages.replace_history_content(loaded)
    assert displayed[0]["content"].startswith("[URL: https://example.test/article]")
    assert inline in displayed[0]["content"] and inline in displayed[1]["content"]
    payload = messages.prepare_chat_messages(loaded)
    assert "Archived page body" in payload[0]["content"][0]["text"]
    assert payload[1]["content"] == "Archived image.\n\n[Generated image]"


def test_missing_asset_degrades_gracefully(history_dir):
    orphan = [{"role": "user", "content": "gone", "image_url": "assets/x/0000.png"}]
    (history_dir / "orphan.json").write_text(json.dumps(orphan), encoding="utf8")

    loaded, _ = history_store.load_history("orphan")
    assert "image_url" not in loaded[0]
    assert loaded[0]["content"] == "gone"


def test_delete_removes_json_and_assets(history_dir):
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    history_store.save_history([{"role": "user", "content": "x", "image_url": inline}], "doomed")
    assert (history_dir / "doomed.json").exists()

    history_store.delete_history("doomed")
    assert not (history_dir / "doomed.json").exists()
    assert not (history_dir / "assets" / "doomed").exists()


def test_select_history_file_reads_the_rendered_rows():
    frame = pd.DataFrame(["alpha", "beta", "gamma"], columns=["File name"])
    assert history_store.select_history_file(frame, 1) == "beta"
    assert history_store.select_history_file(frame, 99) == ""
    assert history_store.select_history_file(frame, None) == ""


def test_scanned_pdf_is_reported_as_scanned(tmp_path):
    """An image-only PDF used to yield the generic 'appears to be empty' warning."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "scanned.pdf"
    with open(path, "wb") as handle:
        writer.write(handle)

    with pytest.raises(readers.ScannedPdfError, match="scanned"):
        readers.read_document(str(path))
