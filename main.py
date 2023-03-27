from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model for brain tumour detection
model = tf.keras.models.load_model('model/brain.h5')

# load the model for medicine prediction
loaded_model = pickle.load(open("model/model.pkl", 'rb'))

# load the medicine dictionary
meds = {0: 'Afinitor (Everolimus),Afinitor Disperz,Alymsys,Avastin', 1: 'Avastin,Belzutifan,BiCNU,Afinitor', 2: 'Belzutifan,Avastin,Temozolomide,Alymsys', 3: 'Everolimus, Danyelza (Naxitamab-gqgk), Naxitamab-gqgk, Temozolomide', 4: 'Alymsys (Bevacizumab), BiCNU (Carmustine),Carmustine, Belzutifan', 5: 'Afinitor (Everolimus), Everolimus, Temozolomide, Lomustine', 6: 'Bevacizumab, Zirabev (Bevacizumab), Temozolomide, Everolimus', 7: 'Everolimus, Zirabev (Bevacizumab), Alymsys (Bevacizumab), Afinitor Disperz (Everolimus)', 8: 'Carmustine,Belzutifan,Alymsys (Bevacizumab),Temodar (Temozolomide)', 9: 'evofloxacin, ceftriaxone, Levaquin, Zithromax', 10: '   clarithromycin, cefdinir, ciprofloxacin, doxycycline', 11: '  Amoxil, Augmentin, Avelox, azithromycin', 12: '   ceftriaxone, doxycycline, azithromycin, Zithromax', 13: '   ciprofloxacin, Levaquin, doxycycline, Augmentin', 14: ' evofloxacin, Amoxil, cefdinir, Zithromax', 15: '  clarithromycin, Avelox, azithromycin, doxycycline', 16: ' Amoxil, ceftriaxone, Avelox, Zithromax', 17: '  evofloxacin, clarithromycin, cefdinir, ciprofloxacin', 18: ' Levaquin, azithromycin, Augmentin, Zithromax', 19: 'Antibiotics, Tamsulosin, Sodium bicarbonate or sodium citrate, Water pills (thiazide diuretics)', 20: 'Allopurinol, Sodium bicarbonate or sodium citrate, Diuretics (water pills), Tamsulosin', 21: ' Diuretics (water pills), Antibiotics, Phosphate solutions, Tamsulosin', 22: 'Water pills (thiazide diuretics), Allopurinol, Sodium bicarbonate or sodium citrate, Diuretics (water pills)', 23: 'Sodium bicarbonate or sodium citrate, Phosphate solutions, Water pills (thiazide diuretics), Tamsulosin', 24: 'Diuretics (water pills), Antibiotics, Allopurinol, Sodium bicarbonate or sodium citrate', 25: 'Tamsulosin, Antibiotics, Water pills (thiazide diuretics), Phosphate solutions', 26: '  Sodium bicarbonate or sodium citrate, Diuretics (water pills), Tamsulosin, Allopurinol', 27: ' Phosphate solutions, Water pills (thiazide diuretics), Antibiotics, Diuretics (water pills)', 28: '    Tamsulosin, Allopurinol, Sodium bicarbonate or sodium citrate, Water pills (thiazide diuretics)'}

# Result classes
def result_class(number):
    if number == 0:
        return 'It is classified as a tumour.'
    else:
        return 'It is not classified as a tumour.'

# Endpoint to predict on an image URL
@app.route('/predictBrainTumor', methods=['GET'])
def predict_brain_tumor():
    print(123)
    # Get the URL parameter from the request
    try:
        url = request.args.get('url')

        print(url)

        # Fetch the image from the URL
        response = requests.get(url)
                
        print(response.status_code)

        # If the response code is 200, the request was successful
        if response.status_code == 200:
            # Access the content of the response as bytes
            image_bytes = response.content
            img = Image.open(BytesIO(image_bytes))
            x = np.array(img.resize((128,128)))
            x = x.reshape(1,128,128,3)
            res = model.predict_on_batch(x)
            classification = np.where(res == np.amax(res))[1][0]
            confidence = str(res[0][classification]*100)
            result = result_class(classification)
            return jsonify({'confidence': confidence, 'result': result})
        else:
            return jsonify({'error': 'Failed to fetch image from URL'}), 400
    except BaseException as e:
        print(e)
        

@app.route('/medicines', methods=['POST'])
def predict_medicine():
    print(request.json)
    age = request.json['age']
    thyroid = request.json['thyroid']
    blood_pressure = request.json['blood_pressure']
    diabetes = request.json['diabetes']
    disease_brain = request.json['disease_brain']
    disease_cancer = request.json['disease_cancer']
    disease_kid = request.json['disease_kid']
    
    result = loaded_model.predict([[age, thyroid, blood_pressure, diabetes, disease_brain, disease_cancer, disease_kid]])
    
    response = {'medicine': meds[result[0]]}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5005)


