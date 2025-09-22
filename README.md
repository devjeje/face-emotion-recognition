realtime-face-emotion-recognition
This project implements a **Real-Time Facial Emotion Recognition System** using **Convolutional Neural Networks (CNN)** built with TensorFlow/Keras and OpenCV.   The model is trained on facial images categorized into seven emotion classes:   `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`.

📌 Features
- Custom CNN model trained on grayscale facial datasets (48x48 pixels).
- Data augmentation to improve model robustness and prevent overfitting.
- Real-time face detection using OpenCV Haar Cascade Classifier.
- Emotion prediction directly from webcam input.
- Live visualization with bounding boxes and emotion labels.

---

🗂️ Project Structure
├── app.py # Main application for real-time emotion recognition
├── train_model.py # Script to train CNN model
├── dataset/ # Dataset directory (train/validation images)
│ ├── train/
│ └── val/
├── model/ # Saved trained model (.h5)
└── README.md # Project documentation


⚙ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/emotion-recognition.git
   cd emotion-recognition
Create a virtual environment (recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🧑‍💻 Usage
1. Train the Model
Make sure your dataset is placed under dataset/train and dataset/val with subfolders for each emotion category.

bash
Copy code
python train_model.py
This will train the CNN and save the best model as:

bash
Copy code
model/emotion_model.h5
2. Run Real-Time Emotion Recognition
bash
Copy code
python app.py
Press q to exit the webcam window.

📊 Model Architecture
The CNN architecture consists of:

3 Convolutional + MaxPooling blocks with Batch Normalization and Dropout.

Fully connected Dense layers with ReLU activation.

Final Softmax layer for multi-class classification.

✅ Requirements
Python 3.7+

TensorFlow / Keras

OpenCV

NumPy

(Alternatively, install via pip install -r requirements.txt)
