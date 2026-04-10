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
        try:
            prompt = request.form.get("prompt")
            style = request.form.get("style")

            # ✅ SAFE CHECK
            if not prompt:
                return render_template("index.html", image_path=None, error="Please enter a prompt")

            if style:
                prompt = f"{prompt}, {style} style"

            # Generate image
            image = client.text_to_image(
                prompt=prompt,
                model="runwayml/stable-diffusion-v1-5"
            )

            # Ensure folder exists
            os.makedirs("static", exist_ok=True)

            image_path = "static/generated.png"
            image.save(image_path)

        except Exception as e:
            print("ERROR:", e)
            return render_template("index.html", image_path=None, error=str(e))

    return render_template("index.html", image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)