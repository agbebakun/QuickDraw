# pip install flask
from flask import Flask, jsonify, request, send_from_directory

from classify import load_model, classify, most_likely, classes
from stroke_to_raster import stroke_to_raster               # For recognising from strokes
from detect_image import normalize_image, load_from_buffer  # For recognising from image data

model = load_model()


app = Flask(__name__)
@app.route('/')
def home():
    # Serve static file
    return send_from_directory('.', 'server.html')


# Web-based recognizer
@app.route('/onnx-recognizer.js')
def onnx_recognizer_js():
    return send_from_directory('.', 'onnx-recognizer.js', mimetype='application/javascript')
@app.route('/ort/ort.min.js')
def ort_min_js():
    return send_from_directory('.', 'ort/ort.min.js', mimetype='application/javascript')
@app.route('/ort/ort-wasm-simd.wasm')
def ort_wasm_simd_wasm():
    return send_from_directory('.', 'ort/ort-wasm-simd.wasm', mimetype='application/wasm')
@app.route('/trained_models/whole_model_quickdraw.onnx')
def onnx_model():
    return send_from_directory('.', 'trained_models/whole_model_quickdraw.onnx', mimetype='application/binary')


# API - return classes
@app.route('/classes.json')
def api_get_classes():
    classes_list = classes()
    return {"classes": classes_list}


# API - classify from sent image file data
@app.route('/classify_image_file', methods=['POST'])
def api_classify_image_file():
    # Get image file from request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']

    # Load image
    file_buffer = file.read()
    image = load_from_buffer(file_buffer)

    # Normalize image
    normalized_image = normalize_image(image)

    # Classify
    class_scores = classify(model, normalized_image)
    detected_class = most_likely(class_scores)

    # Prepare response
    response = {
        "detected_class": detected_class,
        "class_scores": [{ "class": cls, "score": score } for cls, score in class_scores]
    }
    return jsonify(response)


# API - classify from sent image raw data
@app.route('/classify_image_data', methods=['POST'])
def api_classify_image_data():
    # Treat 'image' parameter as base64-encoded raw image data
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    import base64
    image_data_b64 = request.json['image']
    image_data = base64.b64decode(image_data_b64)
    image = load_from_buffer(image_data)
    # Normalize image
    normalized_image = normalize_image(image, debugPrefix='web_image')
    # Classify
    class_scores = classify(model, normalized_image)
    detected_class = most_likely(class_scores)
    # Prepare response
    response = {
        "detected_class": detected_class,
        "class_scores": [{ "class": cls, "score": score } for cls, score in class_scores]
    }
    return jsonify(response)


# API - classify from sent stroke data
@app.route('/classify_strokes', methods=['POST'])
def api_classify_strokes():
    # Get stroke data from request
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400
    data = request.get_json()
    if 'strokes' not in data:
        return jsonify({"error": "No stroke data provided"}), 400
    vector_strokes = data['strokes']

    # Convert strokes to raster image
    raster_image = stroke_to_raster(vector_strokes, debugPrefix='web_strokes')

    # Classify
    class_scores = classify(model, raster_image)
    detected_class = most_likely(class_scores)

    # Prepare response
    response = {
        "detected_class": detected_class,
        "class_scores": [{ "class": cls, "score": score } for cls, score in class_scores]
    }
    return jsonify(response)


host = '127.0.0.1'
port = 5001
app.run(debug=True, host=host, port=port)
