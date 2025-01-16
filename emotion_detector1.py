import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib

# Load dataset
def load_data(dataset_path, image_size=(32, 32), limit_per_class=200):
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    data = []
    target = []
    for label in labels:
        folder_path = os.path.join(dataset_path, label)
        count = 0
        for filename in os.listdir(folder_path):
            if count >= limit_per_class:
                break
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            data.append(img.flatten())
            target.append(labels.index(label))
            count += 1
    return np.array(data), np.array(target)

# Train SVM model with hyperparameter tuning
def train_model(data, target, save_path="svm_emotion_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # SVM pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True))
    ])

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': [0.001, 0.01, 0.1]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model trained with accuracy: {accuracy}')
    
    # Save the trained model
    joblib.dump(best_model, save_path)
    return best_model

# Load pre-trained model
def load_trained_model(model_path="svm_emotion_model.pkl"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# GUI for emotion detection
class EmotionDetector:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Emotion Detector")
        self.root.geometry("400x600")
        self.label = Label(root, text="Upload an image to detect emotion")
        self.label.pack(pady=20)
        self.upload_btn = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)
        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)
        self.panel = None  # To display images

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Preprocess the image for prediction
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))
            img_flattened = img.flatten().reshape(1, -1)  # Match training shape
            
            # Make prediction
            prediction = self.model.predict(img_flattened)
            emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][prediction[0]]
            self.result_label.config(text=f'Emotion Detected: {emotion}')
            
            # Display the uploaded image
            img_display = Image.open(file_path)
            img_display = img_display.resize((200, 200), Image.Resampling.LANCZOS)
            img_display = ImageTk.PhotoImage(img_display)
            if self.panel is None:
                self.panel = Label(self.root, image=img_display)
                self.panel.image = img_display
                self.panel.pack(pady=10)
            else:
                self.panel.config(image=img_display)
                self.panel.image = img_display

if __name__ == "__main__":
    dataset_path = "E:\\Emotion_detector\\dataset"
    model_path = "svm_emotion_model.pkl"

    # Load or train the model
    model = load_trained_model(model_path)
    if not model:
        print("Training model...")
        data, target = load_data(dataset_path, image_size=(32, 32), limit_per_class=500)  # Consistent image size
        model = train_model(data, target, save_path=model_path)
    else:
        print("Loaded pre-trained model.")

    root = Tk()
    app = EmotionDetector(root, model)
    root.mainloop()
