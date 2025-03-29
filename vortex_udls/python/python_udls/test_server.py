from flask import Flask, request, jsonify
import numpy as np
from serialize_utils import WebDataBatcher
# Assume WebDataBatcher is defined in your module or above in the same file

app = Flask(__name__)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    # Get the raw binary data from the request
    data_bytes = request.data
    if not data_bytes:
        return jsonify({'error': 'No data received'}), 400

    # Convert the received bytes into a NumPy array of type uint8
    data_array = np.frombuffer(data_bytes, dtype=np.uint8)
    
    # Create a new WebDataBatcher instance and deserialize the data
    batcher = WebDataBatcher()
    batcher.deserialize(data_array)
    
    # You can now work with your deserialized data
    batch_size = len(batcher.question_ids)
    batcher.print_data()
    return jsonify({'message': 'Data received', 'batch_size': batch_size}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    

