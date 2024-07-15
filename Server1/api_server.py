

from flask import Flask, request, jsonify
import color_analysisFinal  # Import your image processing script
from PIL import Image
app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Process the image using your_script
        # image=Image.open(file.stream())
        img="cropped photo.jpg"
        result = color_analysisFinal.process_image(img)  # Adjust this based on your script
        return jsonify(result)

    return jsonify({'error': 'Something went wrong'}), 500


def allowed_file(filename):
    """Check if the file is an allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
