<div align="center">

# 🧠 NeuralTrust

<!-- Animated Typing Title -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Inter&weight=800&size=40&pause=1000&color=A855F7&center=true&vCenter=true&width=800&lines=NeuralTrust+Diagnostic+System;Multi-Agent+AI+Architecture;Skin+Disease+Classification;Powered+by+Deep+Learning" alt="Typing SVG" /></a>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
</p>

An intelligent, multi-agent artificial intelligence system designed to accurately detect and diagnose various skin conditions using state-of-the-art Convolutional Neural Networks (CNNs).

<br>

![Banner Illustration](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

</div>

## ✨ Features

- 🌌 ** Web Interface:** A gorgeous, dark-themed modern UI featuring dynamic animations and clear, step-by-step pipeline visualization.
-  **Multi-Agent Pipeline:** Utilizes 5 specialized AI agents working in harmony to provide precise diagnostics.
-  **CPU-Optimized:** Designed to run efficiently on consumer hardware without the need for dedicated GPUs.
-  **Intelligent Routing:** Automatically detects the body part (Face, Torso, Limbs) and routes the image to the correct specialist model.
-  **Quality Control:** Built-in vision gatekeeper ensures only high-quality, blur-free, and watermark-free images are processed.
-  **Actionable Guidance:** Breaks down complex medical terms into simple descriptions and provides a clear "What to do" action plan for the user.

<br>

## 🤖 The 5-Agent Architecture

NeuralTrust operates using a sophisticated sequential AI pipeline. Here is the architectural flow:

![Architecture Flowchart](https://mermaid.ink/img/eyJjb2RlIjogImdyYXBoIFREXG4gICAgY2xhc3NEZWYgZGVmYXVsdCBmaWxsOiMxZTFlMjQsc3Ryb2tlOiM1NTUsc3Ryb2tlLXdpZHRoOjFweCxjb2xvcjojZGRkO1xuICAgIFxuICAgIHN1YmdyYXBoIEZyb250ZW5kIFtGcm9udGVuZF1cbiAgICAgICAgVUlbV2ViIFVJIC0gSFRNTC9DU1MvSlNdXG4gICAgZW5kXG5cbiAgICBzdWJncmFwaCBCYWNrZW5kIFtCYWNrZW5kIEZsYXNrXVxuICAgICAgICBBUElbQVBJIFJvdXRlcyAtIGFwcC5weV1cbiAgICAgICAgQTFbQWdlbnQgMTogR2F0ZWtlZXBlcl1cbiAgICAgICAgQTJbQWdlbnQgMjogQm9keSBSb3V0ZXJdXG4gICAgICAgIFxuICAgICAgICBzdWJncmFwaCBTcGVjaWFsaXN0cyBbRGlzZWFzZSBTcGVjaWFsaXN0c11cbiAgICAgICAgICAgIEEzW0FnZW50IDM6IEZhY2lhbF1cbiAgICAgICAgICAgIEE0W0FnZW50IDQ6IFRvcnNvXVxuICAgICAgICAgICAgQTVbQWdlbnQgNTogTGltYnNdXG4gICAgICAgIGVuZFxuICAgIGVuZFxuXG4gICAgc3ViZ3JhcGggU3RvcmFnZSBbQUkgTW9kZWxzXVxuICAgICAgICBXZWlnaHRzW1ByZS10cmFpbmVkIFdlaWdodHNdXG4gICAgZW5kXG5cbiAgICBVSSAtLSBQT1NUIC9kaWFnbm9zZSAtLT4gQVBJXG4gICAgQVBJIC0tIEpTT04gUmVzcG9uc2UgLS0+IFVJXG4gICAgXG4gICAgQVBJIC0tPiBBMVxuICAgIEExIC0tIFBhc3MgLS0+IEEyXG4gICAgQTEgLS4gRmFpbCAuLT4gQVBJXG4gICAgXG4gICAgQTIgLS0gRmFjZSAtLT4gQTNcbiAgICBBMiAtLSBUb3JzbyAtLT4gQTRcbiAgICBBMiAtLSBMaW1icyAtLT4gQTVcbiAgICBcbiAgICBBMyAtLT4gV2VpZ2h0c1xuICAgIEE0IC0tPiBXZWlnaHRzXG4gICAgQTUgLS0+IFdlaWdodHNcbiAgICBcbiAgICBBMyAtLT4gQVBJXG4gICAgQTQgLS0+IEFQSVxuICAgIEE1IC0tPiBBUElcbiIsICJtZXJtYWlkIjogeyJ0aGVtZSI6ICJkYXJrIn19)

| Agent | Name | Technology | Responsibility |
| :---: | :--- | :--- | :--- |
| **1** | 🛡️ **Vision Gatekeeper** | OpenCV | Performs quality checks (sharpness, brightness) and blocks bad images. |
| **2** | 🔀 **Body Part Router** | Haar Cascades | Determines if the image is a Face, Torso, or Limb for proper routing. |
| **3** | 👤 **Facial Specialist** | EfficientNet + ResNet50 | Specialized in facial conditions like Acne and Rosacea. |
| **4** | 🫁 **Torso Specialist** | ResNet34 | Specialized in chest/back conditions like Melanoma and Ringworm. |
| **5** | 🦵 **Limbs Specialist** | ResNet18 | Specialized in arm/leg conditions like Psoriasis and Carcinoma. |

<br>

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system. It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### Running the Application

Start the Flask server by running the following command in your terminal:

```bash
python app.py
```

The application will be live at `http://127.0.0.1:5000`. Open this URL in your web browser to access the NeuralTrust dashboard.

<br>

## 📸 Testing the Project

The system includes a `test_images` directory equipped with specially crafted clinical macro photos for demonstration purposes. To test the pipeline:
1. Open the NeuralTrust Web UI.
2. Click the upload zone.
3. Select any image from the `test_images/` folder (e.g., `1_Acne_Face.png`).
4. Watch the 5-Agent pipeline execute and provide an accurate diagnosis!

---
<div align="center">
  <p>Built with ❤️ for Student Start-up and Innovation Policy (SSIP)</p>
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="500">
</div>
