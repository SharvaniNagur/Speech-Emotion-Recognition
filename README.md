# 🎙️ Speech Emotion Recognition using Attention-Based Deep Learning

## 📌 Overview

This project presents a deep learning framework for Speech Emotion Recognition (SER) using Attention-Based Fully Convolutional Networks (FCN), Long Short-Term Memory (LSTM), and Bidirectional Long Short-Term Memory (BiLSTM) architectures.

The system classifies human speech into four emotional categories:

- 😊 Happy
- 😠 Angry
- 😢 Sad
- 😐 Neutral

The framework combines:
- MFCC feature extraction
- Attention mechanisms
- Deep learning architectures
- Audio signal preprocessing
- Emotion classification
- Performance evaluation using confusion matrices and accuracy curves

---

# 🎯 Problem Statement

Human emotions play a major role in communication and interaction. Traditional systems often fail to accurately identify emotions from speech due to:
- Variations in speech patterns
- Noise
- Pitch changes
- Pronunciation differences

This project improves Speech Emotion Recognition accuracy using attention-based deep learning architectures capable of extracting both spatial and temporal speech features.

---

# 🚀 Objectives

- Develop an intelligent Speech Emotion Recognition framework
- Extract MFCC features from speech signals
- Implement Attention-Based FCN architecture
- Implement Attention-Based LSTM architecture
- Implement Attention-Based BiLSTM architecture
- Compare model performance using evaluation metrics
- Improve classification accuracy using attention mechanisms
- Visualize confusion matrices and training curves

---

# 🗂️ Dataset Information

## 1️⃣ RAVDESS Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) contains professional emotional speech recordings.

### Features
- High-quality emotional speech recordings
- Multiple emotions
- 24 actors
- Sampling frequency: 48 kHz

### Emotions Used
- Happy
- Angry
- Sad
- Neutral

---

## 2️⃣ TESS Dataset

TESS (Toronto Emotional Speech Set) contains emotional speech recordings from professional actresses.

### Features
- High-quality emotional speech samples
- Female voice recordings
- Sampling frequency: 24.4 kHz

### Emotions Used
- Happy
- Angry
- Sad
- Neutral

---

# 📊 Total Dataset Distribution

| Dataset | Audio Files |
|---|---|
| RAVDESS | 669 |
| TESS | 1600 |
| Total | 2269 |

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Neural Network Implementation |
| Librosa | Audio Processing |
| NumPy | Numerical Computation |
| Pandas | Data Handling |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation Metrics |
| Google Colab | Training Environment |

---

# 🔄 Project Workflow

## Step 1: Audio Collection
Speech samples are collected from:
- RAVDESS Dataset
- TESS Dataset

---

## Step 2: Audio Preprocessing

The preprocessing stage includes:
- Noise reduction
- Pre-emphasis filtering
- Framing
- Windowing
- FFT transformation

---

## Step 3: MFCC Feature Extraction

Mel Frequency Cepstral Coefficients (MFCC) are extracted from audio signals.

### Why MFCC?
MFCC captures:
- Pitch
- Frequency information
- Human auditory perception
- Speech timbre

### MFCC Processing Steps
1. Framing
2. Windowing
3. FFT
4. Mel Filter Bank
5. Logarithm
6. Discrete Cosine Transform (DCT)

---

# 🧠 Deep Learning Models

## 1️⃣ Attention-Based Fully Convolutional Network (FCN)

FCN uses convolution layers to extract spatial features from MFCC spectrograms.

### Features
- Learns spatial speech patterns
- Handles variable-length speech signals
- Uses pooling for feature reduction

### Advantages
- Faster training
- Better feature extraction
- Lower complexity

---

## 2️⃣ Attention-Based LSTM

LSTM is a recurrent neural network designed for sequential data processing.

### Features
- Captures temporal dependencies
- Stores long-term information
- Uses memory cells

### LSTM Gates
- Forget Gate
- Input Gate
- Output Gate

### Advantages
- Learns speech sequences effectively
- Retains contextual information

---

## 3️⃣ Attention-Based BiLSTM

BiLSTM processes speech sequences in both forward and backward directions.

### Features
- Captures past and future context
- Improved sequence understanding
- Better contextual learning

### Advantages
- Higher contextual awareness
- Improved emotion detection

---

# 🎯 Attention Mechanism

The attention layer focuses on the most important regions of speech signals.

### Purpose
- Highlight emotionally relevant speech regions
- Improve feature representation
- Increase classification accuracy

### Benefits
- Better learning efficiency
- Improved accuracy
- Enhanced interpretability

---

# 🏗️ Model Architecture

The complete framework consists of:

1. Speech Input
2. Audio Preprocessing
3. MFCC Feature Extraction
4. Attention-Based FCN
5. Attention-Based LSTM
6. Attention-Based BiLSTM
7. Emotion Classification

---

# 📁 Folder Structure

```text
Speech-Emotion-Recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── assets/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   └── training_accuracy.png
│
├── dataset/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   ├── training_fcn.ipynb
│   ├── training_lstm.ipynb
│   └── training_bilstm.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── attention_fcn.py
│   ├── attention_lstm.py
│   ├── attention_bilstm.py
│   └── evaluation.py
│
├── results/
│   ├── confusion_matrices/
│   ├── graphs/
│   └── reports/
│
└── paper/
    └── research_paper.pdf
```

---

# 📄 File Descriptions

## README.md
Contains complete project explanation and documentation.

## preprocessing.py
Performs:
- Noise removal
- Framing
- Windowing
- Signal normalization

## feature_extraction.py
Extracts MFCC features from speech signals.

## attention_fcn.py
Implements Attention-Based Fully Convolutional Network.

## attention_lstm.py
Implements Attention-Based LSTM architecture.

## attention_bilstm.py
Implements Attention-Based Bidirectional LSTM architecture.

## evaluation.py
Computes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# 🏋️ Training Process

1. Load dataset
2. Preprocess audio
3. Extract MFCC features
4. Split dataset
5. Train models
6. Validate models
7. Evaluate performance
8. Generate confusion matrices

---

# 📈 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Correct positive predictions |
| Recall | Detection capability |
| F1-Score | Balance between precision and recall |
| Confusion Matrix | Classification visualization |

---

# 🏆 Results

| Model | Accuracy |
|---|---|
| Attention-Based FCN | 95% |
| Attention-Based LSTM | 92% |
| Attention-Based BiLSTM | 93% |

---

# ✅ Advantages of Proposed System

- Improved accuracy
- Robust emotion classification
- Efficient feature extraction
- Better temporal learning
- Enhanced contextual understanding

---

# 🌍 Applications

Speech Emotion Recognition can be used in:
- Human-Computer Interaction
- Virtual Assistants
- Mental Health Monitoring
- Smart Customer Support
- Automotive Driver Monitoring
- Robotics
- Healthcare Systems

---

# 🔮 Future Improvements

- Real-time emotion recognition
- Multilingual emotion recognition
- Transformer-based architectures
- Audio-visual emotion recognition
- Edge-device deployment

---

# ▶️ How to Run the Project

## Step 1: Clone Repository

```bash
git clone https://github.com/SharvaniNagur/Speech-Emotion-Recognition.git
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Run Training

```bash
python src/attention_fcn.py
```

---

# 📦 Requirements

```text
tensorflow
keras
numpy
pandas
matplotlib
librosa
scikit-learn
```

---

# 🏷️ Repository Topics

Add these GitHub topics:
- speech-emotion-recognition
- deep-learning
- machine-learning
- lstm
- bilstm
- attention-mechanism
- tensorflow
- audio-processing
- mfcc
- emotion-classification

---

# 👩‍💻 Author

**Sharvani V Nagur**  
Electronics and Communication Engineering  
KLE Technological University

---

# 📜 License

This project is licensed under the MIT License.
