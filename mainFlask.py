from flask import Flask, request, jsonify, send_from_directory
import os
from drone_simualted import DroneToDrone, remove_files

image_folder = 'input_images'
output_folder = 'OUTPUT_IMAGES'

app = Flask(__name__)

def initialize_directories_and_d():
    """ Initialize directories and the DroneToDrone object. """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Clear previous files if directory already exists
        remove_files(output_folder)

    # Initialize DroneToDrone object
    global d
    d = DroneToDrone(output_folder=output_folder)
    d.get_input(image_folder)

@app.route('/')
def index():
    return "FLASKFLASKFLASK"

@app.route('/set_number', methods=['POST'])
def set_number():
    data = request.get_json()
    number = data.get('number')
    print(f"Received number: {number}")

    if not number:
        return jsonify({'error': 'No number provided'}), 400

    try:
        # Reinitialize directories and DroneToDrone object
        initialize_directories_and_d()

        # Process the new number
        d.get_number(int(number))
        d.mainProcess()

        print("Number processed and main process executed.")
        return jsonify({'message': 'Number processed successfully'}), 200

    except Exception as e:
        print(f"Error in /set_number endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/output/<filename>')
def output_image(filename):
    if not os.path.exists(output_folder):
        return jsonify({'error': 'Output directory not found'}), 404
    return send_from_directory(output_folder, filename)

@app.route('/data', methods=['GET'])
def get_data():
    try:
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            print(f"Directory {output_folder} does not exist.")
            return jsonify({'error': 'Output directory not found'}), 404

        # Get the list of images
        images = os.listdir(output_folder)
        print(f"Images found: {images}")

        # Get the report (assuming d.report is a list of dictionaries)
        report = d.report
        print(f"Report data: {report}")

        # Combine the data into one JSON response
        return jsonify({
            'images': images,
            'report': report
        })
    except Exception as e:
        print(f"Error in /data endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize directories and DroneToDrone object at startup
    initialize_directories_and_d()
    app.run(host='0.0.0.0', port=5000, debug=True)
