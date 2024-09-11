from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
import io

app = Flask(__name__)

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Molkaatb/Liveness_Vit")
model = AutoModelForImageClassification.from_pretrained("Molkaatb/Liveness_Vit")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate probabilities
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()

    # Map the predicted index to a label
    labels = model.config.id2label
    predicted_label = labels[predicted_class_idx]

    # Prepare response
    response = {
        'prediction': predicted_label,
        'probabilities': {label: prob.item() for label, prob in zip(labels.values(), probabilities[0])}
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
