"""Explicit real LM Studio smoke using synthetic data and disposable storage.

Run: python tests/manual_lmstudio_smoke.py
Requires the local server, a loaded chat model, and an installed embedding model.
Never reads user sources or histories, and never downloads model weights.
"""

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix="lmstudio-smoke-") as temporary:
        root = Path(temporary)
        os.environ["RAG_DATA_DIR"] = str(root / "workspaces")
        os.environ["QDRANT_LOCAL_PATH"] = str(root / "vectors")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        from chatbot import api, config
        from rag_chatbot import service, workspaces
        from rag_chatbot.tools import artifacts
        from rag_chatbot.tools.service import generate_tool

        connection = config.connect("lmstudio", check_embeddings=True)
        with config.use_connection(connection):
            model = config.resolve_model()
            print("Local model:", model, "context:", config.get_max_context_tokens(model))
            print("Embeddings:", config.get_embedding_settings()["model"])
            reply = api.send_chat(model, [{"role": "user", "content": "Reply with only LOCAL_OK."}], temperature=0)
            assert "LOCAL_OK" in reply, reply
            streamed = "".join(api.stream_chat(model, [{"role": "user", "content": "Reply with only STREAM_OK."}], temperature=0))
            assert "STREAM_OK" in streamed, streamed
            workspace = workspaces.create_workspace("Local smoke")
            rag = service.get_service(workspace)
            try:
                source = root / "project.txt"
                source.write_text("Project Cedar launches on June 12. The project budget is 2400 dollars. "
                                  "Mira leads the project. The review meeting is on June 5.", encoding="utf-8")
                for status in rag.ingest([source], "Smoke"):
                    print(status)
                documents = rag.store.list_documents("Smoke")
                assert len(documents) == 1 and documents[0]["status"] == "ready", documents
                ids = [documents[0]["id"]]
                history, references = list(rag.answer("When does Project Cedar launch?", [], "Smoke",
                    document_ids=ids, model=model, stream_output=False, temperature=0))[-1]
                assert references and "12" in history[-1]["content"], history
                print("RAG:", history[-1]["content"])
                result = None
                for event in generate_tool(workspace, "summary", ids, model=model, collection="Smoke",
                        instructions="Write one short section with the launch date, budget and project lead."):
                    print(event["status"])
                    result = event.get("result") or result
                assert result and result["sources"]
                saved = artifacts.save_current_summary(workspace, result)
                restored = artifacts.load_current_summary(workspace)
                assert restored["id"] == saved["id"]
                exported = artifacts.export_summary(workspace, restored, "md", "Local smoke summary")
                assert Path(exported).is_file()
                print("PASS: chat, streaming, embeddings, RAG citations, Summary, save/restore and Markdown export.")
            finally:
                rag.store.close()


if __name__ == "__main__":
    main()
