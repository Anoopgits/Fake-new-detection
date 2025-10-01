FAKE NEW DETECTION:

Fake news spreads quickly across the internet, making it difficult to distinguish between real and fabricated information.
This project leverages Transformers (DistilBERT) and TensorFlow to build a machine learning model that automatically classifies news articles into FAKE or REAL.

It can be integrated into:

News verification systems

Social media monitoring tools

Research on misinformation

Features:

Pretrained DistilBERT for text classification

Built with TensorFlow 2.x / Keras

 upports training on custom datasets

Fast and accurate predictions

Technologies Used :

Programming Language: Python 3.x

Machine Learning Framework: TensorFlow, Keras

NLP Library: Hugging Face Transformers (DistilBERT)

Data Processing: Pandas, NumPy, Scikit-learn

Deployment Options: Streamlit<img width="616" height="835" alt="Screenshot 2025-09-30 233945" src="https://github.com/user-attachments/assets/8c09c25e-9ad3-4761-88a1-fad752fd2884" />


Version Control: Git, GitHub

Screenshots:

<img width="300" height="300" alt="Screenshot 2025-09-30 233945" src="https://github.com/user-attachments/assets/8c09c25e-9ad3-4761-88a1-fad752fd2884" />

    
Project Structure:
Fake-News-Detection/
│── data/                 # Dataset (train/test CSV)
│── model/                # Saved model and tokenizer
│── app.py                # Flask/Streamlit app for deployment
│── train.py              # Training script
│── inference.py          # Prediction script
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation

Setup Instructions:
1️.Clone the Repository
git clone https://github.com/anoop/Fake-News-Detection.git
cd Fake-News-Detection

2️. Create Virtual Environment
python -m venv myenv
source myenv/bin/activate   # Linux/Mac
myenv\Scripts\activate      # Windows

3️. Install Dependencies
pip install -r requirements.txt


requirements.txt:

tensorflow>=2.12
transformers>=4.30
scikit-learn
pandas
numpy
flask
streamlit


Dataset:

Kaggle Fake News Dataset

Any dataset with columns: text, label (0 = Fake, 1 = Real)

Training the Model:

Run:

python train.py


Tokenizes text with DistilBERT tokenizer

Fine-tunes DistilBERT with classification head

Saves trained model and tokenizer in ./model/

Future Improvements:

 Add multimodal detection (text + images)

 Use Explainable AI (XAI) for model interpretability

 Deploy as a REST API with FastAPI

 Integrate into a browser extension

License:

This project is licensed under the MIT License.
You are free to use, modify, and distribute this project with proper attribution.
