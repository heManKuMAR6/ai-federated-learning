import tensorflow as tf
import requests
import json
import numpy as np

# Train a local model (this can be any model training code)
def train_local_model():
    # Create a simple model for the sake of example (replace this with your actual model)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(50,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Simulated training data (use your actual training data here)
    x_train = np.random.rand(1000, 50)
    y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    
    # Train the model (simulated)
    model.fit(x_train, y_train, epochs=2, verbose=1)

    return model

# Convert model weights to JSON serializable format
def serialize_weights(weights):
    return [w.tolist() for w in weights]

# Send weights to the central server
def send_weights_to_central_server(model_weights):
    central_server_url = 'http://<your-central-server-ip>:5000/upload_weights'  # Change this to your server's URL

    # Prepare the data
    data = {
        'weights': serialize_weights(model_weights)
    }

    # Send the weights to the central server
    response = requests.post(central_server_url, json=data)

    if response.status_code == 200:
        print(f"Successfully sent weights to the central server. Response: {response.json()}")
    else:
        print(f"Failed to send weights. Status code: {response.status_code}")

# Main flow
if __name__ == "__main__":
    # Train the local model
    local_model = train_local_model()

    # Get the weights of the trained model
    local_weights = local_model.get_weights()

    # Send weights to the central server
    send_weights_to_central_server(local_weights)
