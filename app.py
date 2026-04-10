import os
from flask import Flask, render_template, request, send_from_directory
from huggingface_hub import InferenceClient

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

def generate_image(prompt):
    if not HF_TOKEN:
        return None, "HF_TOKEN not set"

    try:
        client = InferenceClient(
            provider="nscale",
            api_key=HF_TOKEN,
        )

        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
        )

        path = os.path.join("static", "generated.png")
        image.save(path)

        return path, None

    except Exception as e:
        return None, str(e)


@app.route("/", methods=["GET", "POST"])
def home():
    image_path = None
    error = None

    if request.method == "POST":
        prompt = request.form.get("prompt")

        if prompt:
            image_path, error = generate_image(prompt)

    return render_template("index.html", image_path=image_path, error=error)


if __name__ == "__main__":
    app.run(debug=True)