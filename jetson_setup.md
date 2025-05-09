# Jetson Nano Setup Guide for Vegetable Defect Detection Project

This guide provides step-by-step instructions to set up the Jetson Nano environment required for running the real-time vegetable defect detection system.
I would suggest that you go through instructions from NVIDIA's offical website as well and try using that approach.
In case that does not work and you have similar specs as mine or go through similar problems, you can read through this guide. It contains problems I faced personally and may not apply to everyone.

---

## 📦 Prerequisites

* VVDN Jetson Nano (Model: JN NN by VVDN Technologies)
* microSD card (64GB or higher recommended)
* USB keyboard, mouse, and monitor (HDMI)
* Internet connectivity (Ethernet)
* USB camera

---

## 🖥️ Device Specifications

* **Manufacturer**: VVDN Technologies
* **Model**: VVDN JN NN
* **OS**: Ubuntu 18.04
* **JetPack Version**: 4.6
* **Python**: 3.6 (pre-installed)
* **Important Note**: CUDA, cuDNN, TensorRT, OpenCV, and other deep learning libraries were **not pre-installed** and had to be manually configured.

---

## 🧰 Step 1: Flash JetPack OS

1. Download the [JetPack SDK](https://developer.nvidia.com/embedded/jetpack-sdk-46) and flash it to the SD card using [Balena Etcher](https://www.balena.io/etcher/).
2. Insert the SD card into the Jetson Nano and boot it up.

---

## 🧱 Step 2: Create Python Virtual Environment on SD Card

Due to limited onboard storage, TensorFlow 2.5 was installed in a virtual environment on the SD card:

---

## 🔧 Step 3: Install TensorFlow 2.5 via Manual .whl File

Installing TensorFlow 2.5 on Jetson Nano involved **significant troubleshooting**, including storage errors and dependency conflicts. Most project time was spent resolving these.

> ⚠️ Do not use "pip install tensorflow" directly on Jetson Nano — it requires a special optimized build to leverage hardware acceleration efficiently. Use the recommended .whl file instead.

You can download the exact `.whl` file used (hosted in the GitHub repo):

```bash
pip install tensorflow-2.5.0-cp36-cp36m-linux_aarch64.whl
```

> ⚠️ Dependencies were often installed outside the virtual environment — make sure they are installed into the venv directory as well.

---

## 🔁 Compatibility Adjustments on Laptop

* Training was done on a laptop with Python 3.6 in a virtual environment to ensure TensorFlow compatibility.
* TensorFlow 2.5 installed using the same `.whl` file.
* Numerous **SSL errors** occurred during pip installs, so many packages were manually installed via `.whl` files.

---

## 🧪 Step 5: Training and Inference Workflow

* The model was trained on the laptop using MobileNetV2.
* Exported in TensorFlow’s SavedModel format.
* Model was transferred to the Jetson Nano.
* Inference was run using a live USB camera feed and OpenCV.

---

## 🧯 Troubleshooting Tips

* 💾 Low Storage: Use large-capacity SD cards and install heavy packages in virtual environments.
* 🧩 Dependencies: If pip fails, install `.whl` versions manually.
* 🔒 SSL Errors: Bypass with manual installations or certificate updates.

---
