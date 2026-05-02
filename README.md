<<<<<<< HEAD
# Plant Disease Predictor

A web-based application built with Flask and TensorFlow to predict plant diseases from images. This application allows users to upload a leaf image and receive a prediction along with a confidence score.

## Setup Instructions

These instructions will help you run the project on your local machine.

### Prerequisites

Ensure you have the following installed on your machine:
- Python 3.8 to 3.11 (TensorFlow has specific Python version requirements)
- `pip` (Python package installer)

### Installation

1. **Clone or copy the project folder** to your target machine.
2. **Navigate to the project directory** in your terminal/command prompt:
   ```bash
   cd path/to/Plant-Diseas_Predictor
   ```
3. **Set up a virtual environment** (recommended to avoid dependency conflicts):
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Application

Once the dependencies are installed, you can start the Flask serve.

1. **Ensure your virtual environment is activated** (if you created one).

venv\Scripts\Activate
2. **Run the Flask app:**
   ```bash
   python app.py
   ```
3. **Access the application** in your web browser at:
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

*Note: If you haven't trained a model yet, the app will run in "mock mode" and generate random predictions when you upload an image. See the training section below.*

---

## How to Train the Model

You need to train the model so the application can make real predictions instead of mock ones. We use Transfer Learning with MobileNetV2.

### 1. Prepare Your Dataset

The `train_model.py` script expects your images to be organized in a specific folder structure.

Create a `dataset` folder in the root of your project, with `train` and `valid` subdirectories. Inside these, create folders for each plant disease class:

```
Plant-Diseas_Predictor/
└── dataset/
    ├── train/
    │   ├── Tomato___Healthy/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── Tomato___Late_blight/
    │       ├── img3.jpg
    │       └── img4.jpg
    └── valid/
        ├── Tomato___Healthy/
        │   └── test_img1.jpg
        └── Tomato___Late_blight/
            └── test_img2.jpg
```

*Place your training images in the `train` dataset and a smaller subset (usually 10-20%) in the `valid` dataset.*

### 2. Run the Training Script

Once your dataset is properly organized:

1. **Run the training script:**
   ```bash
   python train_model.py
   ```
2. **Wait for training to complete.** Training time depends on the size of your dataset and your computer's hardware.
3. **Artifacts generated:** Once successful, the script will create a `models` directory containing:
   - `plant_disease_model.keras` (The trained TensorFlow model)
   - `class_indices.json` (A mapping of model output numbers to the actual disease names like 'Tomato___Healthy')

### 3. Restart the App

After training is complete, restart the flask application. The application will automatically detect the newly generated `plant_disease_model.keras` file and start making real predictions!

---

## Transferring the Project to Another Laptop

Transferring this project is simple. Here is a step-by-step guide:

1.  **Stop the running server** (Press `Ctrl+C` in your terminal).
2.  **Zip the Project Folder:** Compress the entire `Plant-Diseas_Predictor` folder into a `.zip` file.
    *   **Tip to save space:** You can delete the `venv` folder before zipping. Virtual environments are specific to the operating system and user paths, so they must be recreated on the new machine anyway. You can also exclude the `dataset` folder if it's very large and you only need to run the pre-trained model. If you want the pre-trained model to work on the new machine, ensure the `models/` directory is included in the zip.
3.  **Transfer the `.zip` file** to the new laptop via USB drive, email, cloud storage (Google Drive, Dropbox), etc.
4.  **Extract the `.zip` file** on the new laptop.
5.  **Follow the "Setup Instructions"** listed above on the new laptop. Because you didn't transfer the `venv` folder, you will need to:
    *   Install Python 3.8+ (if not already installed).
    *   Open terminal/command prompt and navigate to the extracted folder.
    *   Create a new virtual environment: `python -m venv venv`
    *   Activate it: `venv\Scripts\activate` (Windows)
    *   Install dependencies: `pip install -r requirements.txt`
6.  **Run the app:** `python app.py`

If you copied the `models/` folder, the app will work immediately with your trained model. If not, it will run in mock mode until you transfer your dataset and run `python train_model.py` again.
=======
# Plant-Leaf-Disease-Detecting-Model
>>>>>>> d9bcc26bddc21f3ce337f76a0c24520f37eeccb0
