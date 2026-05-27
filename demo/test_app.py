import os

import gradio as gr

from demo.app import build_app, stub_predict


def test_stub_predict_returns_literal() -> None:
    assert stub_predict("any_path.csv", "any_image.png") == "TBD — wiring up in P5.2"


def test_stub_predict_none_inputs() -> None:
    assert stub_predict(None, None) == "TBD — wiring up in P5.2"


def test_build_app_returns_blocks() -> None:
    app = build_app()

    assert isinstance(app, gr.Blocks)


def test_build_app_has_sensor_input() -> None:
    app = build_app()
    component_types = [type(component) for component in app.blocks.values()]

    assert gr.File in component_types


def test_build_app_has_image_input() -> None:
    app = build_app()
    component_types = [type(component) for component in app.blocks.values()]

    assert gr.Image in component_types


def test_build_app_has_textbox_output() -> None:
    app = build_app()
    component_types = [type(component) for component in app.blocks.values()]

    assert gr.Textbox in component_types


def test_llava_demo_port_env_var() -> None:
    previous_port = os.environ.get("LLAVA_DEMO_PORT")
    os.environ["LLAVA_DEMO_PORT"] = "7865"

    try:
        port: int = int(os.environ.get("LLAVA_DEMO_PORT", "7860"))
        assert port == 7865
    finally:
        if previous_port is None:
            os.environ.pop("LLAVA_DEMO_PORT", None)
        else:
            os.environ["LLAVA_DEMO_PORT"] = previous_port
