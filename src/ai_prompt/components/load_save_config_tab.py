import gradio as gr
from gradio.components import Component

from src.ai_prompt.ai_prompt_manager import AiPromptManager
from src.utils import config


def create_load_save_config_tab(ai_prompt_manager: AiPromptManager):
    """
    Creates a load and save config tab.
    """
    input_components = set(ai_prompt_manager.get_components())
    tab_components = {}

    config_file = gr.File(
        label="Load UI Settings from json",
        file_types=[".json"],
        interactive=True
    )
    with gr.Row():
        load_config_button = gr.Button("Load Config", variant="primary")
        save_config_button = gr.Button("Save UI Settings", variant="primary")

    config_status = gr.Textbox(
        label="Status",
        lines=2,
        interactive=False
    )

    tab_components.update(dict(
        load_config_button=load_config_button,
        save_config_button=save_config_button,
        config_status=config_status,
        config_file=config_file,
    ))

    ai_prompt_manager.add_components("load_save_config", tab_components)

    save_config_button.click(
        fn=ai_prompt_manager.save_config,
        inputs=set(ai_prompt_manager.get_components()),
        outputs=[config_status]
    )

    load_config_button.click(
        fn=ai_prompt_manager.load_config,
        inputs=[config_file],
        outputs=ai_prompt_manager.get_components(),
    )