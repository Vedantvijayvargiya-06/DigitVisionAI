# DigitVision AI – Handwritten Digit Recognition

**Minor Project 7 | Student: Vedant Vijay**

DigitVision AI is a handwritten digit recognition application built with a Convolutional Neural Network (CNN) and Streamlit. It uses the MNIST dataset to classify digits from 0 to 9.

## Features

- Upload a handwritten digit image (PNG, JPG, or JPEG) and get a prediction.
- View the predicted digit, confidence score, and class probabilities.
- Explore demo samples from MNIST.
- Review model performance, including evaluation metrics and a confusion matrix.
- Includes the project report PDF.

## Technology stack

Python, TensorFlow/Keras, CNN, Streamlit, NumPy, Matplotlib, Seaborn, Scikit-learn, Pillow.

## Run locally (Windows)

Python 3.11 is recommended.

1. Extract this project folder.
2. Open Command Prompt or PowerShell inside the `DigitVisionAI` folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\\Scripts\\activate.bat`.

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

6. Streamlit will show a local URL, usually `http://localhost:8501`.

The first launch downloads the MNIST dataset and trains the CNN. This can take a few minutes and requires an internet connection. The trained model and metrics are cached in the system temporary directory for subsequent launches.

## Run the CNN workflow in terminal (optional)

```powershell
python app.py --train
```

Optional arguments:

```powershell
python app.py --train --epochs 8 --out part_a_outputs
```

## Deploy on Streamlit Community Cloud

1. Push the contents of this folder to a GitHub repository.
2. Open Streamlit Community Cloud and connect your GitHub account.
3. Create an app using the repository and select `app.py` as the main file.
4. Deploy. The first startup may take longer because the app downloads MNIST and trains the model.

## Project files

- `app.py` – CNN workflow and Streamlit application.
- `requirements.txt` – Python dependencies.
- `VedantVijay_CNN_Streamlit_Report.pdf` – project report.
