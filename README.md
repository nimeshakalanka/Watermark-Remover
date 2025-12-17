# 💧 AI Watermark Remover

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful, local desktop tool that removes watermarks, text, and objects from images using **Deep Learning**. 

Unlike traditional tools that simply blur or smudge the area, this tool uses the **LaMa (Large Mask Inpainting)** model to "hallucinate" and reconstruct the missing background texture, making the removal nearly invisible.

### [Demo: Click Here](https://huggingface.co/spaces/nimeshakalanka24/watermark-remover)

![Demo img](https://res.cloudinary.com/dluwvqdaz/image/upload/v1765707511/Screenshot_2025-12-14_154507_oglluw.png)

## ✨ Features

* **Advanced Inpainting:** Uses State-of-the-Art AI (LaMa) to fill in complex backgrounds.
* **Interactive UI:** Built with Gradio for easy "paint-to-remove" functionality.
* **Privacy First:** Runs 100% locally on your machine. No images are uploaded to the cloud.
* **GPU Accelerated:** Automatically uses NVIDIA GPU if available, falls back to CPU seamlessly.
* **Standalone Mode:** Can be packaged into a single `.exe` file for easy sharing.

## 🛠️ Tech Stack

* **Language:** Python 🐍
* **Core AI:** [PyTorch](https://pytorch.org/)
* **Model:** LaMa (Resolution-robust Large Mask Inpainting)
* **Interface:** [Gradio](https://gradio.app/)
* **Image Processing:** NumPy & Pillow

## 🚀 Installation

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed.

### 2. Clone the Repository
```
git clone [https://github.com/nimeshakalanka/Watermark-Remover.git](https://github.com/nimeshakalanka/Watermark-Remover.git)
cd watermark-remover
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Download the Model
Important: You must download the AI model manually as it is too large for GitHub.

Download big-lama.pt from this link.

Place the file inside the root directory of this project.

💻 Usage
Run the application with:
```
python main.py
```
Wait for the model to load (it will print ✅ Model loaded successfully!).

Your browser will automatically open the local interface.

Upload an image.

Draw over the watermark (be generous with the mask!).

Click Remove Watermark.

📦 Building for Windows (.exe)
You can package this tool into a standalone executable that works without installing Python.

Install PyInstaller:
```
pip install pyinstaller
```
Build the app:

```
pyinstaller --noconfirm --onedir --console --name "WatermarkRemover" main.py
```
Critical Step: Go to the dist/WatermarkRemover folder and copy the big-lama.pt file into it. The app will not work without the model file sitting next to the .exe.
```
📂 Project Structure
├── main.py              # The main application script
├── big-lama.pt          # The AI Model (Download manually)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```
made by ♥️
