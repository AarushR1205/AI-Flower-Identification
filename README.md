# 🌸 Flower Detection AI

A beautiful, interactive Streamlit web app for detecting and classifying flowers using a DenseNet121 deep learning model. Upload a flower image or use your webcam, and the AI will predict the flower type with confidence scores.

---

## 🚀 Features

- **DenseNet121 Model**: Robust, transfer-learned CNN for flower classification.
- **User-Friendly UI**: Modern, responsive interface with custom CSS.
- **Multiple Input Methods**: Upload images or capture directly from your webcam.
- **Confidence Threshold**: Adjustable slider for prediction certainty.
- **Top-3 Predictions**: See the most likely flower types and their probabilities.
- **Tips & Info**: Sidebar with usage tips and model details.

---

## 🏆 Supported Flower Classes

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

---

## 📦 Dataset

- **Source**: [ImageNet Object Localization Challenge (Kaggle)](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)
- **Note**: The model is trained on a subset of ImageNet containing the above flower classes.

---

## 🖥️ Demo

![Demo Screenshot](demo_screenshot.png) <!-- Add your screenshot here -->

---

## 🛠️ Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/flower_detection_model.git
    cd flower_detection_model
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Add the trained model weights**

    - Place your trained DenseNet121 weights file as `densenet121_flower.pth` in the project root.
    - If you don't have a model, [see below](#training-the-model).

4. **Run the app**
    ```bash
    streamlit run app.py
    ```

---

## 📝 Usage

- **Upload Image**: Click "Upload Image", select a flower photo, and click "Analyze Flower".
- **Use Webcam**: Click "Use Webcam", take a photo, and click "Analyze Flower".
- **Adjust Confidence**: Use the sidebar slider to set your desired confidence threshold.
- **View Results**: See the predicted flower type and top-3 predictions with confidence scores.

---

## 🧠 Model Details

- **Architecture**: DenseNet121 (PyTorch, torchvision)
- **Input Size**: 224x224 pixels, RGB
- **Transfer Learning**: Final layer adapted for 5 flower classes
- **Preprocessing**: Standard ImageNet normalization

---

## 🏋️ Training the Model

If you want to train your own model:

1. Download the [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description) dataset.
2. Filter/extract the 5 flower classes.
3. Use PyTorch and torchvision to fine-tune DenseNet121.
4. Save your trained weights as `densenet121_flower.pth`.

*Training code is not included in this repository.*

---

## 📄 Requirements

- Python 3.8+
- streamlit
- torch
- torchvision
- numpy
- pillow
- opencv-python

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
flower_detection_model/
│
├── app.py
├── densenet121_flower.pth
├── requirements.txt
├── demo_screenshot.png
└── README.md
```

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [ImageNet](http://www.image-net.org/)
- [Kaggle](https://www.kaggle.com/)

---

## 📬 License

This project is for educational and research purposes only.

---

## 💡 Tips

- Use clear, well-lit images for best results.
- Make sure the flower is the main subject.
- Try different angles if unsure.

---

Enjoy using **Flower Detection AI**! 🌸