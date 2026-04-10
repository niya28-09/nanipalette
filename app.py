from flask import Flask, render_template, request
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None

    if request.method == "POST":
        prompt = request.form.get("prompt")
        style = request.form.get("style")

        if style:
            prompt += f", {style} style"

        if prompt:
            image = client.text_to_image(
            prompt=prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
)

            image_path = "static/generated.png"
            image.save(image_path)

    return render_template("index.html", image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)