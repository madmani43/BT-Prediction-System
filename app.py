import os
import runpy

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    gradio_script = os.path.join(project_root, "app folder", "app.py")

    if not os.path.isfile(gradio_script):
        raise FileNotFoundError(f"Gradio app not found: {gradio_script}")

    runpy.run_path(gradio_script, run_name="__main__")
