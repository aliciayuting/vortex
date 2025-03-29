import requests
import numpy as np
from serialize_utils import WebDataBatcher

# Generate a dummy batch of 4 samples.
def generate_dummy_batch():
    batcher = WebDataBatcher()
    for i in range(4):
        # Append a unique question ID.
        batcher.question_ids.append(i)
        # Create a dummy image: shape (1, 10, 10) with random pixel values.
        img = np.random.randint(0, 256, size=(1, 10, 10), dtype=np.uint8)
        batcher.images.append(img)
        # Create a dummy question string.
        batcher.questions.append(f"Question {i}")
        # Create a dummy text sequence.
        batcher.text_sequences.append(f"Text sequence {i}")
    return batcher

if __name__ == "__main__":
    # Generate a dummy batch with 4 samples.
    batcher = generate_dummy_batch()
    # Serialize the batch.
    serialized_buffer = batcher.serialize()
    # Convert the NumPy array to a bytes object.
    data_bytes = serialized_buffer.tobytes()
    
    # URL of the server endpoint.
    url = 'http://localhost:5000/upload_data'
    headers = {'Content-Type': 'application/octet-stream'}
    response = requests.post(url, data=data_bytes, headers=headers)
    
    print("Client: Response from server:")
    print(response.status_code)
    print(response.text)