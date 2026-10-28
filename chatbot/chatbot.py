"""Chat-tab callbacks and layout for the local chatbot.

Layout and event wiring only -- model metadata lives in config.py, API calls in
api.py, document parsing in readers.py, token accounting in tokens.py and history
persistence in history_store.py.
"""

from types import SimpleNamespace

import pandas as pd
import gradio as gr

from . import api, config, history_store, readers
from .log import logger, truncate
from .messages import normalize_text, replace_history_content
from .tokens import InputTooLargeError

HISTORY_COLUMNS = ["File name"]


def history_dataframe():
    return pd.DataFrame(history_store.get_history_file_list(), columns=HISTORY_COLUMNS)


def make_reset_inputs():
    """Clear the message and local attachments after a send."""
    return (
        gr.Textbox(label="Message", placeholder="Type a message or question...",
                   autofocus=True, scale=4, value=None),
        gr.Image(label="Upload an image", sources=["upload", "clipboard"],
                 type="filepath", value=None),
        gr.File(label="Upload a document", type="filepath", value=None),
    )


def _keep_inputs():
    """Placeholder for the input components on intermediate streaming yields."""
    return (gr.skip(), gr.skip(), gr.skip())


def _describe_failure(exc):
    """Turn an exception into something worth showing in the UI."""
    if isinstance(exc, InputTooLargeError):
        return str(exc)
    if isinstance(exc, readers.EmptyDocumentError):
        return str(exc)
    if isinstance(exc, RuntimeError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def on_user_input(llm_model, temperature, top_p, text, image, document,
                  history, system_prompt, stream_output):
    """Handle a send. Generator so streamed replies render as they arrive."""
    history = list(history or [])
    prompt_text = normalize_text(text)

    if not (prompt_text or image or document):
        gr.Warning("Enter a message or upload a local document/image before sending.")
        yield (history, replace_history_content(history)) + _keep_inputs()
        return

    try:
        llm_model = config.resolve_model(llm_model)
    except RuntimeError as exc:
        gr.Warning(str(exc))
        yield (history, replace_history_content(history)) + _keep_inputs()
        return

    if image and not config.supports_vision(llm_model):
        gr.Warning(
            f"{llm_model} cannot read images. Connect to a local vision-capable model."
        )
        yield (history, replace_history_content(history)) + _keep_inputs()
        return

    # --- build the user turn -------------------------------------------------
    try:
        user_message = api.build_user_message(
            prompt_text, image_path=image, document_path=document,
        )
    except Exception as exc:
        logger.exception("Failed to build the user message")
        gr.Warning(_describe_failure(exc))
        yield (history, replace_history_content(history)) + _keep_inputs()
        return

    pending_turn_index = len(history)
    history.append(user_message)
    logger.info("User: %s", truncate(user_message["content"], 500))

    # Show the user turn and clear the inputs straight away.
    yield (history, replace_history_content(history)) + make_reset_inputs()

    # --- get the reply -------------------------------------------------------
    try:
        if stream_output:
            history.append({"role": "assistant", "content": ""})
            chunks = []
            stream = api.stream_chat(
                llm_model, history[:-1], temperature=temperature,
                top_p=top_p, system_prompt=system_prompt,
            )
            try:
                for delta in stream:
                    chunks.append(delta)
                    history[-1]["content"] = "".join(chunks)
                    yield (history, replace_history_content(history)) + _keep_inputs()
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()

            history[-1]["content"] = "".join(chunks).strip()
            if not history[-1]["content"]:
                history.pop()
                gr.Warning("The model returned an empty response.")
            yield (history, replace_history_content(history)) + _keep_inputs()

        else:
            ai_message = api.send_chat(
                llm_model, history, temperature=temperature,
                top_p=top_p, system_prompt=system_prompt,
            )
            if ai_message:
                history.append({"role": "assistant", "content": ai_message})
            else:
                gr.Warning("The model returned an empty response.")
            yield (history, replace_history_content(history)) + _keep_inputs()

        logger.info("AI: %s", truncate(history[-1]["content"] if history else "", 500))

    except Exception as exc:
        logger.exception("Request to %s failed", llm_model)
        gr.Warning(_describe_failure(exc))
        # Exclude the failed turn from future API requests, but leave its prompt
        # visible for copying. Restoring input fields here would overwrite any
        # new draft the user typed while the request was running.
        history = history[:pending_turn_index]
        display = replace_history_content(history + [user_message])
        display.append({
            "role": "assistant",
            "content": f"Request failed: {_describe_failure(exc)}\n\n"
                       "The prompt above was not added to the conversation.",
        })
        yield (history, display) + _keep_inputs()


def on_new_chat_click():
    return [], []


def on_toggle_history_column(state):
    state = not state
    return gr.update(visible=state), state


def on_select_history_file(file_list, evt: gr.SelectData):
    row_index = evt.index[0] if evt.index else None
    return history_store.select_history_file(file_list, row_index)


def on_save_history(history, history_file_name):
    try:
        status = history_store.save_history(history, history_file_name)
    except Exception as exc:
        logger.exception("Saving history failed")
        status = f"Error saving history: {_describe_failure(exc)}"
    return status, history_dataframe()


def on_load_history(history_file_name):
    try:
        history, status = history_store.load_history(history_file_name)
        return history, replace_history_content(history), status
    except Exception as exc:
        logger.exception("Loading history failed")
        # A missing/corrupt saved file must not erase the current conversation.
        return gr.skip(), gr.skip(), f"Error loading history: {_describe_failure(exc)}"


def on_delete_history(history_file_name):
    try:
        status = history_store.delete_history(history_file_name)
    except Exception as exc:
        logger.exception("Deleting history failed")
        status = f"Error deleting history: {_describe_failure(exc)}"
    return status, history_dataframe()


def build_tab():
    """Build the Chat tab inside an existing Gradio Blocks/Tab context."""
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                buttons=["copy"], min_height=800,
                # Model replies commonly use these TeX delimiters. Gradio's
                # default enables only $$...$$, leaving inline math unrendered.
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": r"\[", "right": r"\]", "display": True},
                    {"left": r"\(", "right": r"\)", "display": False},
                ],
            )
            state = gr.State([])
            history_column_state = gr.State(True)
            with gr.Row(equal_height=True):
                new_chat_button = gr.Button(value="New chat")
                stop_button = gr.Button(value="Stop", variant="stop")
                toggle_history_column_button = gr.Button(value="Toggle chat history")
        with gr.Column(scale=1) as history_column:
            history_files_list = gr.DataFrame(label="Chat history files",
                                              value=history_dataframe(), max_height=260)
            history_file_name = gr.Textbox(label="File name",
                                           value=history_store.DEFAULT_HISTORY_NAME)
            load_button = gr.Button(value="Load chat history")
            load_status = gr.Markdown(value="")
            save_button = gr.Button(value="Save chat history")
            save_status = gr.Markdown(value="")
            delete_button = gr.Button(value="Delete chat history", variant="stop")
            delete_status = gr.Markdown(value="")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion(label="Prompt"):
                with gr.Row(equal_height=True):
                    text_input = gr.Textbox(label="Message",
                                            placeholder="Type a message or question...",
                                            autofocus=True, scale=4)
                    send_button = gr.Button(value="Send", variant="primary", scale=1,
                                            min_width=80)
                system_prompt = gr.Textbox(label="System prompt",
                                           placeholder="Optional instructions for the AI...",
                                           lines=2)
                llm_model = gr.Dropdown(label="Model", value=config.get_default_model() or None,
                                        choices=config.get_model_choices())
                stream_output = gr.Checkbox(label="Stream responses", value=True)
                temperature = gr.Slider(label="Temperature", minimum=0, maximum=2,
                                        step=0.01, value=1)
                top_p = gr.Slider(label="Top-p", minimum=0, maximum=1, step=0.01, value=1)
        with gr.Column(scale=1):
            with gr.Accordion(label="Input"):
                image_input = gr.Image(label="Upload an image",
                                       sources=["upload", "clipboard"], type="filepath")
                document_input = gr.File(label="Upload a document", type="filepath")

    user_input_inputs = [
        llm_model, temperature, top_p, text_input, image_input, document_input,
        state, system_prompt, stream_output,
    ]
    user_input_outputs = [state, chatbot, text_input, image_input, document_input]

    # Treat Enter and Send as one pending submission, preventing a second model
    # request when both controls are triggered before the first one finishes.
    send_event = gr.on(
        triggers=[text_input.submit, send_button.click],
        fn=on_user_input, inputs=user_input_inputs, outputs=user_input_outputs,
        trigger_mode="once", concurrency_limit=1,
    )
    stop_button.click(None, None, None, cancels=[send_event])

    new_chat_button.click(on_new_chat_click, [], [chatbot, state], cancels=[send_event])
    toggle_history_column_button.click(on_toggle_history_column, history_column_state,
                                       [history_column, history_column_state])

    history_files_list.select(on_select_history_file, history_files_list, history_file_name)
    load_button.click(on_load_history, history_file_name, [state, chatbot, load_status],
                      cancels=[send_event])
    save_button.click(on_save_history, [state, history_file_name],
                      [save_status, history_files_list])
    delete_button.click(on_delete_history, history_file_name,
                        [delete_status, history_files_list])

    return SimpleNamespace(**{name: value for name, value in locals().items()
                              if not name.startswith("_")})
