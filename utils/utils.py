from PIL import Image
import numpy as np
import tensorflow as tf
import os

def is_valid_fingerprint(image_path):
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return None
        
        # Open the image using PIL
        image = Image.open(image_path)

        # Check if the image is in RGB format, if not, convert it
        image = image.convert('RGB')  # Ensure the image is in RGB format
        
        # Resize the image to match the input size required by the model (224x224 in this example)
        image = image.resize((224, 224))

        # Convert the image to a numpy array
        image_array = np.array(image) / 255.0  # Normalize the image

        # Print the shape of the image array for debugging
        print(f"Image array shape: {image_array.shape}")  # Expected output: (224, 224, 3)
        
        # Add a batch dimension to the image array (model expects a batch)
        image_array = np.expand_dims(image_array, axis=0)

        # Check the shape after adding the batch dimension
        print(f"Image array shape after adding batch dimension: {image_array.shape}")  # Expected output: (1, 224, 224, 3)

        # Load your model (adjust the path to your model)
        model = tf.keras.models.load_model(r'C:\Users\Vasundra\Downloads\deep\models\fingerprint_bloodgroup_model_densenet.h5')

        # Make a prediction
        prediction = model.predict(image_array)

        # Check the prediction
        print(f"Prediction: {prediction}")

        # Get the index of the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map the predicted class index to the corresponding blood group
        blood_groups = ['O+', 'A+', 'B+', 'AB+']  # Update this list based on your model's classes
        return blood_groups[predicted_class]

    except Exception as e:
        print(f"Error processing image: {e}")
        return None
