from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np

def load_trained_model(model_path):
    # Load and return the trained model
    return load_model(model_path)

def predict_blood_group(model, img_path, class_indices):
    # Load the image and resize it to the model input size
    img = load_img(img_path, target_size=(224, 224))
    
    # Convert image to array and normalize the pixel values
    img_array = img_to_array(img) / 255.0
    
    # Expand dimensions to match the input shape of the model (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Reverse the class indices dictionary to map the predicted index to the class label
    class_labels = {v: k for k, v in class_indices.items()}
    
    # Return the predicted class label
    return class_labels[predicted_class_index]
