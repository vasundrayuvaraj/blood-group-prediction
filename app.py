from flask import Flask, flash, redirect, render_template, request, session, url_for
import os
from utils.nutrition import nutritions_recommend
from utils.predict import load_trained_model, predict_blood_group
from werkzeug.utils import secure_filename

from utils.compatibility import compatibility_checking


# type: ignore


# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using flash messages

# Configure app for file uploads
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Update ALLOWED_EXTENSIONS to include bmp
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Allowed file extensions function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load trained model
model_path = r'C:\Users\Vasundra\Downloads\deep\models\fingerprint_bloodgroup_model_densenet.h5'  # Correct path
model = load_trained_model(model_path)
CLASS_INDICES = {'A-': 0, 'A+': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    uploaded_image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        # Save the uploaded file to the output folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Set the image URL to display on the page
        uploaded_image_url = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        # Predict the blood group
        prediction_result = predict_blood_group(model, filepath, CLASS_INDICES)

    return render_template('predict.html', prediction_result=prediction_result, uploaded_image_url=uploaded_image_url)

@app.route('/services')
def services():
    return render_template('services.html')


@app.route("/nutrition",methods=['GET','POSt'])
def nutrition():
    # blood_group=request.args.get('blood_group')
    blood_group = session.get('blood_group')
    return render_template("nutrition.html",blood_group=blood_group)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/developers')
def developers():
    return render_template('developers.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Handle image upload logic here
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # type: ignore
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('File uploaded successfully!')
        else:
            flash('Invalid file type or no file selected.')
        return redirect(url_for('upload_image'))

    return render_template('upload_image.html')

import os
from flask import Flask, request, redirect, url_for, session, flash, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import cv2


# Function to preprocess fingerprint image for model input
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to model's expected input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Function to get blood group from the fingerprint image
def get_blood_group(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    for blood_group, idx in CLASS_INDICES.items():
        if idx == predicted_class:
            return blood_group
    return None

# Function to check compatibility between donor and recipient blood groups
def check_compatibility(donor_blood_group, recipient_blood_group):
    compatibility = {
        'A+': ['A+', 'AB+'],
        'A-': ['A-', 'A+', 'AB+', 'AB-'],
        'B+': ['B+', 'AB+'],
        'B-': ['B-', 'B+', 'AB+', 'AB-'],
        'AB+': ['AB+'],
        'AB-': ['AB-', 'AB+', 'A-', 'B-'],
        'O+': ['O+', 'A+', 'B+', 'AB+'],
        'O-': ['O-', 'O+', 'A-', 'B-', 'AB-', 'AB+']
    }

    if recipient_blood_group in compatibility.get(donor_blood_group, []):
        return True
    return False

@app.route('/transfusion', methods=['GET', 'POST'])
def transfusion():
    if request.method == 'POST':
        # Handle form submission
        return redirect(url_for('handle_donor'))
    return render_template('transfusion.html')

# Handle donor blood group
@app.route('/handle_donor', methods=['POST'])
def handle_donor():
    blood_group = request.form.get('donor_blood_group')
    fingerprint = request.files.get('donor_fingerprint')
    
    # Ensure blood group is valid and formatted correctly
    if blood_group:
        blood_group = blood_group.strip().upper()  # Clean and normalize the input
        
        # Check if the blood group is valid (e.g., 'O+', 'A-', etc.)
        valid_blood_groups = ['A-', 'A+', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        if blood_group not in valid_blood_groups:
            flash('Invalid blood group. Please enter a valid blood group.')
            return redirect(url_for('transfusion'))
        
        session['donor_blood_group'] = blood_group
        return redirect(url_for('transfusion'))
    
    # If fingerprint is provided
    if fingerprint:
        filename = secure_filename(fingerprint.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fingerprint.save(file_path)
        
        # Check if the fingerprint is valid
        is_valid = is_valid_fingerprint(file_path)  # Assuming you have a function to check fingerprint validity
        if is_valid:
            blood_group = get_blood_group(file_path)  # Get the blood group from fingerprint prediction
            session['donor_blood_group'] = blood_group
            return redirect(url_for('transfusion'))
        else:
            flash('Unable to classify the image. Please upload a correct fingerprint image.')
            return redirect(url_for('transfusion'))
    
    flash('Please provide either a blood group or a fingerprint image.')
    return redirect(url_for('transfusion'))

# Handle recipient blood group
@app.route('/handle_recipient', methods=['POST'])
def handle_recipient():
    blood_group = request.form.get('recipient_blood_group')
    fingerprint = request.files.get('recipient_fingerprint')
    
    # Ensure blood group is valid and formatted correctly
    if blood_group:
        blood_group = blood_group.strip().upper()  # Clean and normalize the input
        
        # Check if the blood group is valid (e.g., 'O+', 'A-', etc.)
        valid_blood_groups = ['A-', 'A+', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        if blood_group not in valid_blood_groups:
            flash('Invalid blood group. Please enter a valid blood group.')
            return redirect(url_for('transfusion'))
        
        session['recipient_blood_group'] = blood_group
        return redirect(url_for('transfusion'))
    
    # If fingerprint is provided
    if fingerprint:
        filename = secure_filename(fingerprint.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fingerprint.save(file_path)
        
        # Check if the fingerprint is valid
        is_valid = is_valid_fingerprint(file_path)  # Assuming you have a function to check fingerprint validity
        if is_valid:
            blood_group = get_blood_group(file_path)  # Get the blood group from fingerprint prediction
            session['recipient_blood_group'] = blood_group
            return redirect(url_for('transfusion'))
        else:
            flash('Unable to classify the image. Please upload a correct fingerprint image.')
            return redirect(url_for('transfusion'))
    
    flash('Please provide either a blood group or a fingerprint image.')
    return redirect(url_for('transfusion'))


@app.route('/check_compatibility', methods=['POST'])
def check_compatibility():
    donor_blood_group = session.get('donor_blood_group')
    recipient_blood_group = session.get('recipient_blood_group')

    # Debugging: Print the types and values of the session variables
    print(f"Donor Blood Group: {donor_blood_group}, Type: {type(donor_blood_group)}")
    print(f"Recipient Blood Group: {recipient_blood_group}, Type: {type(recipient_blood_group)}")

    # Check if either donor or recipient blood group is missing
    if donor_blood_group is None or recipient_blood_group is None:
        flash("Both donor and recipient blood groups are required to check compatibility.")
        return redirect(url_for('transfusion'))  # Redirect back to transfusion page if missing

    # Ensure both blood groups are strings. If they are tuples, join them.
    if isinstance(donor_blood_group, tuple):
        donor_blood_group = ''.join(donor_blood_group)  # Join tuple like ('O', '+') -> 'O+'

    if isinstance(recipient_blood_group, tuple):
        recipient_blood_group = ''.join(recipient_blood_group)  # Join tuple like ('O', '+') -> 'O+'

    # Check if donor_blood_group and recipient_blood_group are strings
    if isinstance(donor_blood_group, str) and isinstance(recipient_blood_group, str):
        compatibility_result = compatibility_checking(donor_blood_group, recipient_blood_group)
    else:
        flash("Invalid data type for blood group. Please try again.")
        print(f"Invalid data detected: Donor: {donor_blood_group}, Recipient: {recipient_blood_group}")
        return redirect(url_for('transfusion'))  # Handle invalid data type

    # Check compatibility result and render appropriate page
    if compatibility_result[0] == "Compatible":
        return render_template('compatibility.html', result="Compatible", description=compatibility_result[1])
    else:
        return render_template("compatibility.html", result="Not Compatible", description="")






@app.route('/reset_donor', methods=['POST'])
def reset_donor():
    session.pop('donor_blood_group',None)
    return redirect(url_for('transfusion'))

@app.route('/reset_recipient', methods=['POST'])
def reset_recipient():
    session.pop('recipient_blood_group',None)
    return redirect(url_for('transfusion'))
    
@app.route('/nutrition_recommendation', methods=['POST'])
def nutrition_recommendation():
    # Retrieve the blood group from session
    blood_group = session.get('blood_group')
    
    # Debugging print statement
    print(f"Blood Group from Session: {blood_group}")
    
    if blood_group is None:
        return redirect(url_for('nutrition'))
    
    # Check if blood_group is a tuple and join the elements
    if isinstance(blood_group, tuple):
        blood_group = ''.join(blood_group)  # Join the tuple into a string (e.g. ('O', '+') -> 'O+')
    
    # Ensure blood group is in full format
    if blood_group == 'O':
        blood_group = 'O+'  # Set to O+ if just 'O' is passed

    # Get the nutritional recommendations based on the blood group
    recommendations = nutritions_recommend(blood_group)
    
    if recommendations is None:
        return render_template('recommendation.html', blood_group=blood_group, foods_to_take=[], foods_to_avoid=[])
    
    # Ensure blood group is in uppercase
    blood_group = blood_group.upper()
    
    # Split the recommendations
    foods_to_take = recommendations[0].split(',')
    foods_to_avoid = recommendations[1].split(',')
    
    return render_template('recommendation.html', blood_group=blood_group, foods_to_take=foods_to_take, foods_to_avoid=foods_to_avoid)



@app.route('/handle_input', methods=['POST'])
def handle_input():
    blood_group = request.form.get('blood_group')
    fingerprint = request.files.get('fingerprint')
    
    if fingerprint:
        filename = secure_filename(fingerprint.filename) # type: ignore
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fingerprint.save(file_path)
        is_valid=is_valid_fingerprint(file_path)
        if is_valid:
            blood_group = get_blood_group(file_path)
        else:
            return redirect(url_for('nutrition'))
    
    if blood_group:
        session['blood_group'] = blood_group
        return redirect(url_for('nutrition'))
    
    flash('Please provide a valid input.')
    return redirect(url_for('nutrition'))

@app.route('/reset_input', methods=['POST'])
def reset_input():
    session.pop('blood_group', None)
    return redirect(url_for('nutrition'))

@app.route('/predict_nutrition',methods=['GET','POST'])
def predict_nutrition():
    blood_group = request.args.get('blood_group')
    session['blood_group']=blood_group
    return redirect(url_for('nutrition'))

#predict_transfusion
@app.route('/predict_transfusion',methods=['GET','POST'])
def predict_transfusion():
    
    if request.method == 'POST':
        blood_group = request.form['blood_group']
        session['donor_blood_group'] = blood_group
        return redirect(url_for('transfusion'))
    else:
        blood_group=request.args.get('blood_group')
        session['donor_blood_group'] = blood_group
        return redirect(url_for('transfusion'))
    

@app.route('/predict_result/<filename>',methods=['GET','POST'])
def prediction(filename):
    return render_template('prediction.html', filename=filename)


def is_valid_fingerprint(image_path):
    if model_predict_fc(image_path,model) == 0: # type: ignore
        return 0
    return 1
    

def get_blood_group(image_path):
    return is_valid_fingerprint(image_path)
 # type: ignore

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore



# Step 3: Define the nutrition recommendations
nutrition_recommendations = {
    'A+': {
        'foods_to_take': ["Lean meats", 
                          "Vegetables", 
                          "Fruits", 
                          "Whole grains"],
        'foods_to_avoid': ["Processed meats", 
                           "Dairy products", 
                           "Wheat-based foods"]
    },
    'A-': {
        'foods_to_take': ["Fruits", "Vegetables", "Legumes", "Nuts"],
        'foods_to_avoid': ["Dairy products", "Meat", "Wheat-based foods"]
    },
    'B+': {
        'foods_to_take': ["Lean meats", "Dairy products", "Green vegetables", "Rice"],
        'foods_to_avoid': ["Processed meats", "Wheat-based foods"]
    },
    'B-': {
        'foods_to_take': ["Lean meats", "Dairy products", "Green vegetables", "Rice"],
        'foods_to_avoid': ["Wheat-based foods", "Corn"]
    },
    'O+': {
        'foods_to_take': ["Meat",
                           "Fish", 
                           "Vegetables", 
                           "Nuts"],
        'foods_to_avoid': ["Dairy products", 
                           "Grains"]
    },
    'O-': {
        'foods_to_take': ["Lean proteins", "Vegetables", "Fruit"],
        'foods_to_avoid': ["Grains", "Dairy products"]
    },
    'AB+': {
        'foods_to_take': ["Animal proteins", "Plant-based foods", "Fruits", "Vegetables"],
        'foods_to_avoid': ["Highly processed foods", "Sugary foods"]
    },
    'AB-': {
        'foods_to_take': ["Lean proteins", "Vegetables", "Fruits"],
        'foods_to_avoid': ["Processed foods", "Wheat-based foods"]
    }
}


# Step 4: Preprocess the image for prediction
def load_and_preprocess_image(image_path):
    # Load the image with target size matching the model's input shape
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust to your model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image to [0, 1] range if required
    return img_array

# Step 5: Predict blood group using the trained model
def model_predict_fc(img_array, model):
    # Make a prediction (assuming the model outputs a probability for each class)
    prediction = model.predict(img_array)
    
    # Get the index of the highest probability (for multi-class classification)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class
    
    # Get the blood group corresponding to the predicted class index
    blood_group = list(CLASS_INDICES.keys())[predicted_class_index]
    
    # Ensure blood_group is a string (just to be safe)
    blood_group = str(blood_group)
    
    return blood_group  # Return the blood group as a string



@app.route('/predict', methods=['POST', 'GET'])
def predict1():
    if request.method == 'POST':
        # Assuming you're handling the image upload and prediction here
        image = request.files['image']  # Get uploaded image
        img_path = image.filename
        image.save(img_path)  # Save the image file (optional)

        # Preprocess the image and predict the blood group
        img_array = load_and_preprocess_image(img_path)
        blood_group = model_predict_fc(img_array, model)
        
        # Get nutrition recommendations using the blood group string
        nutrition = nutritions_recommend(blood_group)
        
        # Pass blood group and nutrition separately to the template
        return render_template('recommendations.html', blood_group=blood_group, nutrition=nutrition)
    
    return render_template('recommendations.html')

@app.route('/recommendations', methods=['GET'])
def recommendations():
    blood_group = request.args.get('blood_group')  # Get the blood group from the URL parameter

    # Get the nutrition recommendations based on the blood group
    if blood_group in nutrition_recommendations:
        nutrition = nutrition_recommendations[blood_group]
        foods_to_take = nutrition['foods_to_take']
        foods_to_avoid = nutrition['foods_to_avoid']
    else:
        foods_to_take = []
        foods_to_avoid = []

    # Render the recommendations page and pass the blood group and food recommendations
    return render_template('recommendation.html', blood_group=blood_group, 
                           foods_to_take=foods_to_take, foods_to_avoid=foods_to_avoid)




# Step 6: Main function to validate fingerprint and predict blood group
def is_valid_fingerprint(image_path):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)
    
    # Get the prediction and nutrition recommendations from the model
    blood_group, nutrition = model_predict_fc(img_array, model)
    
    # Return the result
    return blood_group, nutrition


if __name__ == '__main__':
    app.run(debug=True)
