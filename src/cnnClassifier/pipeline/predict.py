import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.class_labels = [
            'adho mukha svanasana', 'adho mukha vriksasana', 'agnistambhasana',
            'ananda balasana', 'anantasana', 'anjaneyasana', 'ardha bhekasana',
            'ardha chandrasana', 'ardha matsyendrasana',
            'ardha pincha mayurasana', 'ardha uttanasana',
            'ashtanga namaskara', 'astavakrasana', 'baddha konasana',
            'bakasana', 'balasana', 'bhairavasana', 'bharadvajasana i',
            'bhekasana', 'bhujangasana', 'bhujapidasana', 'bitilasana',
            'camatkarasana', 'chakravakasana', 'chaturanga dandasana',
            'dandasana', 'dhanurasana', 'durvasasana',
            'dwi pada viparita dandasana', 'eka pada koundinyanasana i',
            'eka pada koundinyanasana ii', 'eka pada rajakapotasana',
            'eka pada rajakapotasana ii', 'ganda bherundasana',
            'garbha pindasana', 'garudasana', 'gomukhasana', 'halasana',
            'hanumanasana', 'janu sirsasana', 'kapotasana', 'krounchasana',
            'kurmasana', 'lolasana', 'makara adho mukha svanasana',
            'makarasana', 'malasana', 'marichyasana i', 'marichyasana iii',
            'marjaryasana', 'matsyasana', 'mayurasana', 'natarajasana',
            'padangusthasana', 'padmasana', 'parighasana',
            'paripurna navasana', 'parivrtta janu sirsasana',
            'parivrtta parsvakonasana', 'parivrtta trikonasana',
            'parsva bakasana', 'parsvottanasana', 'pasasana',
            'paschimottanasana', 'phalakasana', 'pincha mayurasana',
            'prasarita padottanasana', 'purvottanasana', 'salabhasana',
            'salamba bhujangasana', 'salamba sarvangasana',
            'salamba sirsasana', 'savasana', 'setu bandha sarvangasana',
            'simhasana', 'sukhasana', 'supta baddha konasana',
            'supta matsyendrasana', 'supta padangusthasana', 'supta virasana',
            'tadasana', 'tittibhasana', 'tolasana', 'tulasana',
            'upavistha konasana', 'urdhva dhanurasana', 'urdhva hastasana',
            'urdhva mukha svanasana', 'urdhva prasarita eka padasana',
            'ustrasana', 'utkatasana', 'uttana shishosana', 'uttanasana',
            'utthita ashwa sanchalanasana', 'utthita hasta padangusthasana',
            'utthita parsvakonasana', 'utthita trikonasana', 'vajrasana',
            'vasisthasana', 'viparita karani', 'virabhadrasana i',
            'virabhadrasana ii', 'virabhadrasana iii', 'virasana',
            'vriksasana', 'vrischikasana', 'yoganidrasana'
        ]
        self.model = load_model(os.path.join(
            "artifacts", "training", "model.h5"))
        # self.model = load_model(os.path.join("model", "model.h5"))

    def predict(self):
        try:
            # Preprocess the image
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            # Rescale the image to the [0, 1] range
            test_image = test_image / 255.0

            # Predict the class of the image
            prediction = self.model.predict(test_image)
            result = np.argmax(prediction, axis=1)

            # Map the prediction result to the corresponding class label
            predicted_label = self.class_labels[result[0]]
            return {"prediction": predicted_label}
        except Exception as e:
            return {"error": str(e)}
