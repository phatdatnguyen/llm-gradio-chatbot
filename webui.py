"""Application shell: each chatbot owns its own layout, events, and state."""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from chatbot import config
from chatbot.chatbot import build_tab as build_chat_tab
from chatbot.log import setup_logging
from rag_chatbot.chatbot import build_tab as build_rag_tab
from rag_chatbot.chatbot import on_prepare_connection_change
from plugins.ui import build_tab as build_plugins_tab
from chatbot.connections import (build_tab as build_connections_tab, connect_connection,
                                 request_connection, release_session, refresh_connected_library)


APP_CSS = """
.rag-source-checkboxes [data-testid="checkbox-group"] {
    max-height: 260px;
    overflow-y: auto;
}
.rag-source-checkboxes [data-testid="checkbox-group"] > label {
    width: 100%;
    box-sizing: border-box;
}
.rag-source-checkboxes [data-testid="checkbox-group"] > label > span {
    overflow-wrap: anywhere;
    min-width: 0;
}
.rag-summary-sidebar {
    justify-content: space-between;
}
.rag-summary-preview {
    overflow-y: auto;
}
.rag-quiz-options .wrap {
    flex-direction: column;
    flex-wrap: nowrap;
}
.rag-quiz-options .wrap > label {
    width: 100%;
    box-sizing: border-box;
}
.rag-quiz-options .wrap > label > span {
    overflow-wrap: anywhere;
    min-width: 0;
}
.rag-quiz-options input[type="radio"] {
    flex-shrink: 0;
}
"""

# Prevent generated or saved Markdown/HTML from fetching external media. The
# Gradio frontend, bundled fonts, event stream and local previews use this origin.
LOCAL_HEAD = """<meta http-equiv="Content-Security-Policy" content="default-src 'self' data: blob:;
script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:;
style-src 'self' 'unsafe-inline'; connect-src 'self';
img-src 'self' data: blob:; media-src 'self' data: blob:; font-src 'self' data:;
object-src 'none';">"""


def create_app():
    config.set_connection_resolver(request_connection)
    with gr.Blocks(title="Local Chatbots", analytics_enabled=False) as app:
        with gr.Tabs():
            with gr.Tab("Connections"):
                connections_tab = build_connections_tab()
            with gr.Tab("Chat"):
                chat_tab = build_chat_tab()
            with gr.Tab("RAG Chat"):
                rag_tab = build_rag_tab()
            with gr.Tab("Plugins"):
                plugins_tab = build_plugins_tab()
        app.load(plugins_tab["refresh_callback"],
                 outputs=[plugins_tab["catalog"], plugins_tab["status"],
                          plugins_tab["snapshot"]], queue=False)
        connection_inputs = [connections_tab.connector, chat_tab.llm_model, rag_tab.model,
                             rag_tab.operation_state, connections_tab.embedding_connector,
                             connections_tab.textgen_context]
        connection_outputs = [connections_tab.status, chat_tab.llm_model, rag_tab.model]
        prepare_connection = connections_tab.connect.click(on_prepare_connection_change,
            [rag_tab.history_state, rag_tab.pending_state, rag_tab.workspace_state, rag_tab.operation_state],
            rag_tab.answer_outputs, queue=False, cancels=[rag_tab.send_event])
        connect_event = prepare_connection.then(connect_connection, inputs=connection_inputs,
                 outputs=connection_outputs, concurrency_limit=None, trigger_mode="always_last")
        connect_event.then(refresh_connected_library,
            [rag_tab.workspace_state, rag_tab.collection_filter, rag_tab.document_filter,
             rag_tab.review_document, rag_tab.operation_state], rag_tab.library_outputs,
            preprocess=False, concurrency_id="rag-workspace-storage", concurrency_limit=1)
        app.unload(release_session)
    app.chat_tab = chat_tab
    app.rag_tab = rag_tab
    app.plugins_tab = plugins_tab
    app.connections_tab = connections_tab
    return app


demo = create_app()

if __name__ == "__main__":
    logger = setup_logging()
    config.check_model_tables()
    logger.info("Starting disconnected; choose a local connector and press Connect.")
    demo.queue().launch(server_name="127.0.0.1", share=False,
                        max_file_size=100 * gr.FileSize.MB, css=APP_CSS, head=LOCAL_HEAD)
