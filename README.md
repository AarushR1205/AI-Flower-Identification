# 📝 AI Flower Identifier Project Report

## 1\. Executive Summary

The **AI Flower Identifier** project is a web application built using **Streamlit** that leverages a pre-trained **DenseNet-201** Convolutional Neural Network (CNN) in **PyTorch** to classify flower images into one of 102 distinct species. The application features a highly aesthetic, vibrant user interface with custom CSS, a robust PyTorch-based inference pipeline, and a key feature allowing users to set a dynamic **Confidence Threshold** to filter low-certainty predictions. The model loading logic includes a critical fix to address state dictionary key mismatches commonly encountered when customizing pre-trained models.

## 2\. Technical Architecture & Components

| Component | Technology/Tool | Role in Project |
| :--- | :--- | :--- |
| **Frontend/App Framework** | Streamlit (`app.py`) | Provides the interactive web interface, handles file uploads, and displays results. |
| **Core ML Framework** | PyTorch (`torch`, `torchvision`) | Used for loading, inference, and management of the DenseNet-201 model. |
| **Model** | DenseNet-201 | The CNN architecture used for feature extraction and classification across 102 flower categories. |
| **Pre-processing** | `torchvision.transforms` | Implements the standard ImageNet pre-processing pipeline (Resize, CenterCrop, ToTensor, Normalize) to prepare input images (224x224) for the model. |
| **Class Mapping** | JSON (`cat_to_name.json`) | Maps the model's output indices (1-102) to human-readable flower names. |
| **Dependencies** | `requirements.txt` | Lists all necessary Python packages and their required versions (e.g., `streamlit`, `torch`, `Pillow`). |

## 3\. Key Implementation Details

### 3.1. PyTorch Model Loading & State Dictionary Fix

The `load_resources()` function handles the critical steps of loading the pre-trained `densenet201` model, replacing its classifier head for 102 classes, and loading the saved state dictionary (`densenet201_best_model.pth`).

A major point of failure addressed was a **key mismatch** in the state dictionary. When a simple `nn.Linear` layer is assigned directly to `model.classifier`, the expected keys are `classifier.weight` and `classifier.bias`. However, the saved model likely stored the classifier as a `nn.Sequential` (e.g., during training), leading to keys like `classifier.0.weight`.

**The fix implemented:**

```python
# KEY MAPPING FIX: Rename keys from 'classifier.0.weight' to 'classifier.weight'
# and 'classifier.0.bias' to 'classifier.bias' to solve the error.
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('classifier.0.'):
        new_key = k.replace('classifier.0.', 'classifier.')
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
```

This ensures the saved weights are correctly mapped to the initialized model architecture.

### 3.2. Prediction and Output

The `predict_flower` function performs inference:

1.  **Image Preprocessing:** The input image is converted to a PyTorch tensor and normalized.
2.  **Inference:** `model(processed_img)` is called under `torch.no_grad()`.
3.  **Probability Calculation:** `F.softmax` is applied to the raw logits to get probabilities.
4.  **Top-K Selection:** `np.argsort` is used to efficiently retrieve the **top 5** predicted classes and their corresponding confidence scores.

### 3.3. Dynamic Confidence Threshold

A Streamlit **sidebar slider** allows the user to set a minimum confidence score (default **80%**).

  * If the top prediction's confidence is **equal to or above** the threshold, a vibrant, large result card is displayed.
  * If the confidence is **below** the threshold, a clear, yellow **low-confidence warning card** is displayed instead, ensuring the user is not misled by a potentially poor prediction.

### 3.4. User Interface and Experience (UX)

The application heavily utilizes custom CSS to achieve a **vibrant, animated, and modern design**. Key elements include:

  * Gradient background with a subtle animation (`gradientShift`).
  * Prominent, glowing header text (`glow`).
  * Animated, border-enhanced upload area.
  * Detailed result cards with colorful gradient confidence bars and shimmer effects (`shimmer`, `gradientFlow`).

### 3.5. File Upload Update

The file uploader type list was updated to include support for the modern **WEBP** image format, increasing the application's versatility:

```python
uploaded_file = st.file_uploader(
    "Choose an image",
    type=['jpg', 'jpeg', 'png', 'webp'], # 🌟 Added 'webp' 🌟
    label_visibility="collapsed"
)
```

-----

-----

# 🌺 AI Flower Identifier (PyTorch)

This is a Streamlit web application that uses a pre-trained **DenseNet-201** Convolutional Neural Network (CNN), implemented in **PyTorch**, to classify flower images into one of 102 species.

## ⚙️ Project Structure

For the application to run correctly, ensure your project directory has the following structure:

```
flower-identifier/
├── app.py              # The main Streamlit application code
├── cat\_to\_name.json    # The mapping from class index to flower name
├── requirements.txt    # List of required Python packages
└── densenet201\_best\_model.pth # **Your trained PyTorch model file**
```

**Note:** You must place your trained PyTorch model file, saved as a state dictionary, in the root directory and name it **`densenet201_best_model.pth`**.

-----

## 🚀 Setup and Run

Follow these steps to set up and run the application locally.

### 1\. Install Dependencies

Install all required packages using `pip` and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2\. Run the Application

Start the Streamlit application from your terminal:

```bash
streamlit run app.py
```

This command will open the application in your default web browser (usually at `http://localhost:8501`).

-----

## ✨ Application Features

  * **PyTorch Backend:** Utilizes the standard **`torch`** and **`torchvision.models.densenet201`** for high-performance image classification.
  * **Aesthetic UI:** Features a vibrant, modern, and animated Streamlit user interface built with custom CSS.
  * **Robust Image Support:** Accepts **JPG, JPEG, PNG, and WEBP** image formats.
  * **Custom Prediction Display:**
      * The highest-confidence prediction (Rank 1) is featured prominently in a large card.
      * The remaining predictions (Ranks 2-5) are listed separately under "Other Top Predictions."
  * **Adjustable Confidence Threshold:** Includes a **sidebar slider** to set a minimum confidence level for the main prediction card. If the highest prediction falls below this threshold, a warning message is displayed instead of the main result, promoting result accuracy.
  * **Model Loading Fix:** Includes crucial logic to resolve key mismatches when loading the state dictionary of the modified DenseNet-201 classifier.
  * **Image Preprocessing:** Implements the standard PyTorch ImageNet preprocessing pipeline (`Resize`, `CenterCrop`, `ToTensor`, `Normalize`) to correctly prepare the input image for the DenseNet-201 model.

-----

## 🔍 Model Details

  * **Architecture:** DenseNet-201
  * **Classification:** 102 different flower categories.
  * **Input Size:** Images are resized and centrally cropped to 224x224 pixels.