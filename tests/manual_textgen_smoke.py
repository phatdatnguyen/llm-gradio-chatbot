"""Optional real textgen/Ollama smoke using synthetic data and temporary storage.

Run explicitly: python tests/manual_textgen_smoke.py
Requires textgen with a loaded supported llama.cpp model and Ollama with an
installed embedding model. Selects Ollama embeddings explicitly; never downloads
models, reads user documents, or changes server loading/settings. The app's default
textgen context remains 4096 unless TEXTGEN_CONTEXT_LENGTH is already configured.
"""

import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def progress(message):
    print(message, flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="textgen-smoke-") as temporary:
        root = Path(temporary)
        environment = {"RAG_DATA_DIR": str(root / "workspaces"),
                       "QDRANT_LOCAL_PATH": str(root / "vectors"),
                       "GRADIO_ANALYTICS_ENABLED": "False"}
        with patch.dict(os.environ, environment):
            from chatbot import api, config
            from rag_chatbot import service, workspaces
            from rag_chatbot.tools import artifacts
            from rag_chatbot.tools.service import generate_tool

            progress("Connecting to textgen with explicitly selected Ollama embeddings.")
            connection = config.connect("textgen", embedding_connector="ollama", check_embeddings=True)
            rag = None
            try:
                with config.use_connection(connection):
                    model = config.resolve_model()
                    embedding = config.get_embedding_settings()
                    progress(f"Chat model: {model}; effective context: {config.get_max_context_tokens(model)}.")
                    progress(f"Embedding model: {embedding['model']}; dimensions: {embedding['dimensions']}.")
                    assert not connection.embedding_error, connection.embedding_error
                    assert embedding["model"] and embedding["dimensions"] > 0
                    assert embedding["identity"].startswith("ollama:"), embedding["identity"]
                    assert config.supports_structured_output(model), "This textgen loader does not support Studio generation."

                    progress("Checking ordinary chat with native prompt preflight.")
                    reply = api.send_chat(model, [{"role": "user", "content": "Reply with only LOCAL_OK."}],
                                          temperature=0)
                    assert "LOCAL_OK" in reply, reply
                    progress("Checking streaming chat.")
                    streamed = "".join(api.stream_chat(model,
                        [{"role": "user", "content": "Reply with only STREAM_OK."}], temperature=0))
                    assert "STREAM_OK" in streamed, streamed

                    progress("Checking the explicitly selected embedding facade.")
                    vectors = api.embed_texts([embedding["query_prefix"] + "Synthetic local smoke check."],
                                             embedding["model"], embedding["dimensions"])
                    assert len(vectors) == 1 and len(vectors[0]) == embedding["dimensions"]

                    workspace = workspaces.create_workspace("Textgen smoke")
                    rag = service.get_service(workspace)
                    source = root / "project.txt"
                    source.write_text("Project Cedar launches on June 12. The project budget is 2400 dollars. "
                                      "Mira leads the project. The review meeting is on June 5.", encoding="utf-8")
                    progress("Indexing one synthetic source into temporary embedded Qdrant.")
                    for status in rag.ingest([source], "Smoke"):
                        progress(status)
                    documents = rag.store.list_documents("Smoke")
                    assert len(documents) == 1 and documents[0]["status"] == "ready", documents
                    ids = [documents[0]["id"]]

                    progress("Checking a grounded RAG answer and its local citation.")
                    history, references = list(rag.answer("When does Project Cedar launch?", [], "Smoke",
                        document_ids=ids, model=model, stream_output=False, temperature=0))[-1]
                    answer = history[-1]["content"]
                    assert references and "June 12" in answer, answer
                    assert any(f"[{reference['number']}]" in answer for reference in references), answer
                    assert history[-1]["sources"] == references
                    progress("RAG answer: " + answer)

                    progress("Generating a schema-constrained Summary through native GBNF.")
                    result = None
                    for event in generate_tool(workspace, "summary", ids, model=model, collection="Smoke",
                            instructions="Write one short section with the launch date, budget and project lead."):
                        progress(event["status"])
                        result = event.get("result") or result
                    assert result and result["sources"] and result["coverage"]["processed_chunks"] > 0
                    saved = artifacts.save_current_summary(workspace, result)
                    restored = artifacts.load_current_summary(workspace)
                    assert restored == saved
                    exported = artifacts.export_summary(workspace, restored, "md", "Textgen smoke summary")
                    assert Path(exported).is_file() and Path(exported).stat().st_size > 0
                    assert r"project\.txt" in Path(exported).read_text(encoding="utf-8")
                    progress("Summary save, restore, and Markdown export passed.")
                    for tool_id, key, settings in (
                        ("quizzes", "questions", {"item_count": 2}),
                        ("presentation", "slides", {"slide_count": 2}),
                    ):
                        progress(f"Checking {tool_id} with exact counts and scoped grammar constraints.")
                        generated = None
                        for event in generate_tool(workspace, tool_id, ids, model=model, collection="Smoke",
                                instructions="Use only the project launch date, budget, lead and review date. Keep the result concise.",
                                **settings):
                            progress(event["status"])
                            generated = event.get("result") or generated
                        assert generated and len(generated["content"][key]) == 2
                    progress("Vision skipped: textgen does not expose verified loaded vision capabilities.")
                    progress("PASS: textgen chat, streaming, Ollama embeddings, RAG citations, Summary, Quizzes, Presentation, and persistence/export.")
            finally:
                if rag is not None:
                    rag.store.close()
                config.release_connection(connection)


if __name__ == "__main__":
    main()
