# Real-Time Defective Vegetable Detection using Jetson Nano

A lightweight deep learning system for **real-time classification of fresh vs. stale vegetables** using MobileNetV2, deployed on NVIDIA Jetson Nano. This edge-AI solution enables offline, low-latency inference for smart agriculture and automated food quality control.

> ⚠️ **Note:** This project is still in progress and the model is not fully optimized. Current inference speed and robustness are sufficient for demonstrations but **may not be production-ready**. Optimization steps like TensorRT conversion, pruning, quantization, and better dataset augmentation are recommended before deployment. 

---

## 📌 Features

* 🧠 MobileNetV2 pre-trained and fine-tuned on custom dataset
* 🎥 Live classification via USB camera and OpenCV overlay
* 💡 Handles lighting variations, occlusions, and camera angle changes
* ⚡ Real-time inference on Jetson Nano (<2s latency per frame)
* 📦 TensorFlow 2.5 and Python 3.6 optimized for edge deployment

---

## 📽️ Demo

![demo\_screenshot](assets/demo_screenshot.png)
![sample\_results](assets/sample_results.gif)

---

## 🛠️ Architecture

![architecture\_diagram](assets/architecture_diagram.png)

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/vegetable-defect-detection.git
cd vegetable-defect-detection
pip install -r requirements.txt
```

### Jetson Nano Setup (Optional)

Refer to [jetson\_setup.md](jetson_setup.md) for setting up Python 3.6 + TensorFlow 2.5 on Jetson Nano.

---

## 🧪 Usage

### Model Training

```bash
python src/model_training.py
```

### Real-Time Inference on Jetson

```bash
python src/inference_jetson.py
```

---

## 📂 Project Structure

```
vegetable-defect-detection/
├── src/                  # Python source files
├── data/                 # Training and test images
├── saved_model/          # Exported model
├── assets/               # Diagrams, GIFs, screenshots
├── notebooks/            # Jupyter notebooks (optional)
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
```

---

## 📊 Dataset

This project uses a dataset sourced from [Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification). Special thanks to the dataset creator:

> **Dataset Name**: *Fresh and Stale Classification*
> **Creator**: *Swoyam*
> [View Dataset on Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)
> *(Used under the terms of the dataset’s license)*

---

## 📈 Results

* ✅ Accuracy: **>85%** on test set
* ⚡ Inference time: \~1–2 seconds per frame
* 🧪 Tested under variable lighting and occlusion conditions

---

## 🚧 Limitations & Future Work

* 📷 Sensitive to extreme lighting or unfamiliar fruit types
* 🔄 No on-device learning (training is offline)
* ⏱️ Model is not yet optimized for deployment; potential improvements:

  * Convert to TensorRT or TensorFlow Lite for faster inference
  * Apply model pruning and quantization to reduce size
  * Train on larger and more diverse datasets
  * Explore batch inference and camera input filtering for stability

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🔖 Tags

`#JetsonNano` `#EdgeAI` `#ComputerVision` `#TensorFlow` `#MobileNetV2` `#Python` `#AgriTech` `#OpenCV`

