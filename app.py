from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests

# Flask Uygulaması
app = Flask(__name__)

# CLIP Modelini Yükle
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    if 'image_url' not in request.json:
        return jsonify({"error": "Görsel URL'si eksik"}), 400

    image_url = request.json['image_url']

    try:
        # Görseli indir ve PIL Image olarak aç
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Embedding hesaplama
        inputs = processor(images=image, return_tensors="pt", padding=True)
        outputs = model.get_image_features(**inputs)
        embedding = outputs.detach().numpy().tolist()

        return jsonify({"embedding": embedding}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
