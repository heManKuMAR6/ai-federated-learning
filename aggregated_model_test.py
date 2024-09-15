import tensorflow as tf
import numpy as np

# Load the aggregated model
def load_aggregated_model(model_path='aggregated_model.h5'):
    """
    Load the saved aggregated model from the central server.
    """
    return tf.keras.models.load_model(model_path)

# Function to test the loaded model
def test_model(model, input_shape=(50,)):
    """
    Test the model with a dummy input (or real data).
    Here, we use a random input for demonstration purposes.
    """
    # Generate random input data
    test_input = np.random.rand(1, *input_shape)  # Example input with shape (1, 50)
    
    # Perform prediction
    predictions = model.predict(test_input)
    
    print("Test input:", test_input)
    print("Predicted output:", predictions)
    print("Predicted class:", np.argmax(predictions))

if __name__ == '__main__':
    # Path to the aggregated model
    model_path = 'aggregated_model.h5'
    
    # Load the aggregated model
    model = load_aggregated_model(model_path)
    
    # Test the model
    test_model(model)
