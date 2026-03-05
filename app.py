from flask_cors import CORS
from flask import Flask, request, jsonify
import torch
import os

from models.cnn3d import CNN3D   # import your model

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model architecture
model = CNN3D().to(device)

# Load trained weights
model.load_state_dict(torch.load(
    "models/best_model.pth", map_location=device))

model.eval()


def preprocess_mri(path):

    volume = torch.load(path)

    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0)

    volume = volume.unsqueeze(0)

    return volume


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    os.makedirs("uploads", exist_ok=True)

    filepath = os.path.join("uploads", file.filename)

    file.save(filepath)

    volume = preprocess_mri(filepath).to(device)

    with torch.no_grad():

        outputs = model(volume)

        probs = torch.softmax(outputs, dim=1)

        confidence = float(torch.max(probs))

        pred = torch.argmax(outputs, 1).item()

    if pred == 0:
        result = "Non Demented"
    else:
        result = "Alzheimer Detected"

    return jsonify({
        "result": result,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
