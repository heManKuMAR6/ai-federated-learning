import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your central model (ensure this is the same architecture as the local models)
model = create_model()  # Make sure create_model is defined and matches the local model structure

# List to store weights received from clients
received_weights = []

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    global received_weights
    data = request.json

    # Deserialize received weights
    client_weights = [np.array(w) for w in data['weights']]
    received_weights.append(client_weights)

    return jsonify({"message": "Weights received successfully!", "total_hits": len(received_weights)}), 200

@app.route('/aggregate', methods=['GET'])
def aggregate_weights():
    global received_weights

    if len(received_weights) == 0:
        return jsonify({"message": "No weights to aggregate."}), 400

    # Initialize list to store average weights
    averaged_weights = []

    # Number of clients that have submitted weights
    num_clients = len(received_weights)

    # Check if the length of each set of received weights matches the central model
    if not all(len(client_weights) == len(model.get_weights()) for client_weights in received_weights):
        return jsonify({"message": "Mismatch in number of weights between clients and central model."}), 400

    # Average the weights across all clients
    for weights_list_tuple in zip(*received_weights):
        averaged_weights.append(np.mean(weights_list_tuple, axis=0))

    # Debugging: print out the dimensions of the central model's weights and the averaged weights
    central_model_weights = model.get_weights()
    print(f"Central Model Weights Length: {len(central_model_weights)}")
    print(f"Averaged Weights Length: {len(averaged_weights)}")

    for i, (central_weight, avg_weight) in enumerate(zip(central_model_weights, averaged_weights)):
        print(f"Layer {i}: Central model weight shape: {central_weight.shape}, Averaged weight shape: {avg_weight.shape}")

    # Set the averaged weights to the central model
    try:
        model.set_weights(averaged_weights)
    except ValueError as e:
        return jsonify({"message": f"Error setting weights: {str(e)}"}), 500

    # Save the aggregated model
    model.save('aggregated_model.h5')
    
    # Reset the received weights for next round
    received_weights = []

    return jsonify({"message": "Weights aggregated successfully and model saved as aggregated_model.h5", "total_aggregations": 1}), 200
