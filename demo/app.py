"""Gradio UI shell for the llava-for-sensors demo (P5.1).

Real inference wiring is in P5.2 (VOI-214). This module exposes:
  - stub_predict(sensor_csv_path, image_path) -> str
  - build_app() -> gr.Blocks
  - main() -- launches the server
"""

import os

import gradio as gr


def stub_predict(sensor_csv_path: str | None, image_path: str | None) -> str:
    """Stub inference handler. Returns a literal placeholder string.

    Args:
        sensor_csv_path: Path to the uploaded sensor CSV file, or None.
        image_path: Path to the uploaded bearing image, or None.

    Returns:
        Literal stub string; replaced by real inference in P5.2.
    """
    return "TBD — wiring up in P5.2"


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks UI without launching a server.

    Returns:
        A gr.Blocks instance defining the demo UI.
    """
    with gr.Blocks(title="LLaVA-for-Sensors Demo") as demo:
        gr.Markdown("## LLaVA-for-Sensors: Bearing Fault Prediction")
        gr.Markdown(
            "Upload a sensor CSV and a bearing image to get a fault prediction "
            "and rationale."
        )
        with gr.Row():
            sensor_input = gr.File(
                label="Sensor CSV",
                file_types=[".csv"],
                type="filepath",
            )
            image_input = gr.Image(
                label="Bearing Image",
                type="filepath",
            )
        predict_btn = gr.Button("Predict")
        output = gr.Textbox(
            label="Prediction & Rationale",
            lines=5,
            interactive=False,
        )
        predict_btn.click(
            fn=stub_predict,
            inputs=[sensor_input, image_input],
            outputs=[output],
        )
    return demo


def main() -> None:
    """Launch the Gradio server.

    Reads LLAVA_DEMO_PORT from the environment (default: 7860).
    Binds to 127.0.0.1 only; never shares publicly.
    """
    port: int = int(os.environ.get("LLAVA_DEMO_PORT", "7860"))
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=port, share=False)


if __name__ == "__main__":
    main()
