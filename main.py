import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('brain_tumor_cnn_model.h5')

# Preprocess the image


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150)
                         )  # Same size as used in training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Same normalization as in training
    return img_array


# Load and preprocess the image for prediction
img_path = 'test.jpg'
img_array = preprocess_image(img_path)

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Output the predicted class
classes = ['glioma', 'meningioma', 'no tumor',
           'pituitary']  # Replace with your class names
print(f"Predicted Class: {classes[predicted_class[0]]}")
