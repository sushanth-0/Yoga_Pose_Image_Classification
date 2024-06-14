from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

# Ensure correct environment variables for language and locale
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("dvc repro")  # Trigger DVC to reproduce the pipeline
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']  # Get the image from the request
        decodeImage(image, clApp.filename)  # Decode the image
        result = clApp.classifier.predict()  # Get the prediction
        return jsonify(result)  # Return the prediction as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    clApp = ClientApp()  # Initialize the client app
    app.run(host='0.0.0.0', port=8081)  # Run the Flask app
