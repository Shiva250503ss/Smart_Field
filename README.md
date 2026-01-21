<div align="center">

# 🌾 Smart Field
### *Cultivating Tomorrow's Harvest with AI & Data Science*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)

<img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/homepage.jpg" alt="Smart Field Banner" width="800"/>

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Shiva250503ss-181717?style=flat-square&logo=github)](https://github.com/Shiva250503ss)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shivaraj_Senthil_Rajan-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shivaraj-senthil-rajan-2b8898227/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit_Site-00C7B7?style=flat-square&logo=netlify)](https://shiva250503ss.github.io/shivaraj-portfolio/)
[![Email](https://img.shields.io/badge/Email-CU_Boulder-CFB87C?style=flat-square&logo=gmail)](mailto:Shivaraj.SenthilRajan@colorado.edu)

---

**An end-to-end Machine Learning solution for precision agriculture, combining Deep Learning image classification, predictive analytics, and real-time IoT data integration to empower data-driven farming decisions.**

[Features](#-core-features) • [Tech Stack](#-technology-stack) • [ML Models](#-machine-learning-architecture) • [Demo](#-application-demo) • [Installation](#-quick-start)

</div>

---

## 🎯 Project Highlights

<table>
<tr>
<td width="50%">

### 🔬 Data Science & ML
- **Deep Learning** model with ResNet9 architecture
- **38-class image classification** for plant diseases
- **Random Forest** predictive modeling for crop recommendations
- **Feature Engineering** with 7 environmental parameters
- **Real-time API integration** for weather data

</td>
<td width="50%">

### 📊 Data Engineering & Analytics
- **ETL Pipeline** for agricultural data processing
- **RESTful API** architecture with Flask
- **Structured data management** with Pandas
- **IoT sensor data integration** capabilities
- **Production deployment** with Gunicorn WSGI

</td>
</tr>
</table>

---

## 🛠️ Technology Stack

<div align="center">

### 🤖 Machine Learning & AI
| Category | Technologies |
|:--------:|:------------|
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![TorchVision](https://img.shields.io/badge/TorchVision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| **Machine Learning** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Random Forest](https://img.shields.io/badge/Random_Forest-228B22?style=flat-square) |
| **Computer Vision** | ![PIL](https://img.shields.io/badge/Pillow-3776AB?style=flat-square&logo=python&logoColor=white) ![Image Processing](https://img.shields.io/badge/Image_Processing-FF6F00?style=flat-square) |

### 📊 Data Engineering & Analytics
| Category | Technologies |
|:--------:|:------------|
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Data Serialization** | ![Pickle](https://img.shields.io/badge/Pickle-3776AB?style=flat-square&logo=python&logoColor=white) ![JSON](https://img.shields.io/badge/JSON-000000?style=flat-square&logo=json&logoColor=white) |
| **API Integration** | ![REST API](https://img.shields.io/badge/REST_API-009688?style=flat-square) ![OpenWeatherMap](https://img.shields.io/badge/OpenWeatherMap-EB6E4B?style=flat-square&logo=openweathermap&logoColor=white) |

### 🌐 Backend & Deployment
| Category | Technologies |
|:--------:|:------------|
| **Framework** | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white) ![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=flat-square&logo=jinja&logoColor=white) |
| **Server** | ![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=flat-square&logo=gunicorn&logoColor=white) ![Heroku](https://img.shields.io/badge/Heroku-430098?style=flat-square&logo=heroku&logoColor=white) |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white) ![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=flat-square&logo=bootstrap&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black) |

</div>

---

## 🧠 Machine Learning Architecture

### 1️⃣ Plant Disease Detection — Deep Learning CNN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🔬 ResNet9 Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input Image (256x256x3)                                                    │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐               │
│   │   Conv Block  │───▶│ Residual Block│───▶│   Conv Block  │               │
│   │   64 filters  │    │   128 filters │    │  256 filters  │               │
│   └───────────────┘    └───────────────┘    └───────────────┘               │
│         │                                          │                         │
│         │              ┌───────────────┐           │                         │
│         └─────────────▶│ Residual Block│◀──────────┘                         │
│                        │  512 filters  │                                     │
│                        └───────┬───────┘                                     │
│                                │                                             │
│                                ▼                                             │
│                    ┌─────────────────────┐                                   │
│                    │   Classification    │                                   │
│                    │    (38 Classes)     │                                   │
│                    └─────────────────────┘                                   │
│                                                                              │
│   Features: BatchNorm • ReLU • MaxPooling • Residual Connections            │
└─────────────────────────────────────────────────────────────────────────────┘
```

<details>
<summary><b>🌿 Supported Disease Classes (38 Classes across 14 Crops)</b></summary>

| Crop | Diseases Detected |
|:----:|:------------------|
| 🍎 **Apple** | Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🍇 **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍅 **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| 🥔 **Potato** | Early Blight, Late Blight, Healthy |
| 🌽 **Corn** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍒 **Cherry** | Powdery Mildew, Healthy |
| 🫑 **Pepper** | Bacterial Spot, Healthy |
| 🍊 **Orange** | Haunglongbing (Citrus Greening) |
| 🍑 **Peach** | Bacterial Spot, Healthy |
| 🫐 **Blueberry** | Healthy |
| 🍓 **Strawberry** | Leaf Scorch, Healthy |
| 🎃 **Squash** | Powdery Mildew |
| 🌱 **Soybean** | Healthy |
| 🍇 **Raspberry** | Healthy |

</details>

---

### 2️⃣ Crop Recommendation — Random Forest Classifier

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    📊 Feature Engineering Pipeline                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│   │  Nitrogen   │    │ Phosphorous │    │  Potassium  │                      │
│   │     (N)     │    │     (P)     │    │     (K)     │                      │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                      │
│          │                  │                  │                              │
│          └──────────────────┼──────────────────┘                              │
│                             │                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │    Soil     │    │  Rainfall   │    │ Temperature │    │  Humidity   │   │
│   │     pH      │    │    (mm)     │    │   (API) 🌐  │    │  (API) 🌐   │   │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│          │                  │                  │                  │           │
│          └──────────────────┴──────────────────┴──────────────────┘           │
│                                     │                                         │
│                                     ▼                                         │
│                    ┌─────────────────────────────┐                            │
│                    │   🌳 Random Forest Model   │                            │
│                    │     (Ensemble Learning)     │                            │
│                    └──────────────┬──────────────┘                            │
│                                   │                                           │
│                                   ▼                                           │
│                    ┌─────────────────────────────┐                            │
│                    │   🌾 Optimal Crop Prediction │                           │
│                    │      (22+ Crop Types)       │                            │
│                    └─────────────────────────────┘                            │
│                                                                               │
│   🌐 Real-Time Weather Data: OpenWeatherMap API Integration                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Supported Crops:** Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

### 3️⃣ Fertilizer Recommendation — Rule-Based Expert System

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    🧪 Nutrient Analysis Engine                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   User Input                  Reference Database              Decision Logic  │
│   ┌─────────────┐            ┌─────────────────┐            ┌─────────────┐  │
│   │ Current NPK │───────────▶│  fertilizer.csv │───────────▶│  Comparator │  │
│   │   Values    │            │ (Optimal Values)│            │    Engine   │  │
│   └─────────────┘            └─────────────────┘            └──────┬──────┘  │
│                                                                     │         │
│                              ┌──────────────────────────────────────┘         │
│                              │                                                │
│                              ▼                                                │
│   ┌───────────────────────────────────────────────────────────────────────┐  │
│   │                    Recommendation Categories                          │  │
│   ├───────────────────┬───────────────────┬───────────────────────────────┤  │
│   │  N-High / N-Low   │  P-High / P-Low   │      K-High / K-Low          │  │
│   │  ↓                │  ↓                │      ↓                       │  │
│   │  Organic/Synthetic│  Bone Meal/       │      Wood Ash/               │  │
│   │  Alternatives     │  Rock Phosphate   │      Potash Fertilizers      │  │
│   └───────────────────┴───────────────────┴───────────────────────────────┘  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Smart_Field/
│
├── 📂 Code/
│   ├── 🐍 app.py                     # Flask REST API & ML Inference Engine
│   ├── ⚙️ config.py                  # Environment Configuration
│   ├── 📋 requirements.txt           # Dependencies
│   ├── 🚀 Procfile                   # Heroku Deployment Config
│   │
│   ├── 📂 models/                    # Trained ML Models
│   │   ├── 🧠 plant_disease_model.pth    # PyTorch CNN (26.4 MB)
│   │   └── 🌳 RandomForest.pkl           # Scikit-learn Model (697.8 KB)
│   │
│   ├── 📂 utils/                     # Utility Modules
│   │   ├── 🔧 model.py               # ResNet9 Architecture Definition
│   │   ├── 📖 disease.py             # Disease Information Database
│   │   └── 🧪 fertilizer.py          # Fertilizer Recommendations
│   │
│   ├── 📂 Data/                      # Data Assets
│   │   └── 📊 fertilizer.csv         # Crop Nutrient Requirements
│   │
│   ├── 📂 templates/                 # Jinja2 HTML Templates
│   │   ├── 🏠 index.html             # Landing Page
│   │   ├── 🌾 crop.html              # Crop Recommendation Form
│   │   ├── 🔬 disease.html           # Disease Detection Upload
│   │   └── 🧪 fertilizer.html        # Fertilizer Analysis Form
│   │
│   └── 📂 static/                    # Frontend Assets
│       ├── 🎨 css/                   # Bootstrap + Custom Styles
│       ├── 📜 scripts/               # JavaScript Modules
│       └── 🖼️ images/                # UI Assets
│
├── 📂 documents/                     # Project Documentation
│   ├── 📑 FINAL YEAR PROJECT REPORT_SF.pdf
│   └── 📑 Smart_Fields_Review_Paper_final1.pdf
│
└── 📂 output screenshots/            # Demo Screenshots
```

---

## 📸 Application Demo

<div align="center">

### 🏠 Homepage & Services

<table>
<tr>
<td align="center"><b>Landing Page</b></td>
<td align="center"><b>Our Services</b></td>
</tr>
<tr>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/homepage.jpg" width="400"/></td>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/services.jpg" width="400"/></td>
</tr>
</table>

---

### 🔬 Plant Disease Detection (Deep Learning)

<table>
<tr>
<td align="center"><b>📤 Image Upload</b></td>
<td align="center"><b>📊 Disease Analysis Result</b></td>
</tr>
<tr>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/plant_disease_input.jpg" width="400"/></td>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/plant_disease_output.jpg" width="400"/></td>
</tr>
</table>

---

### 🌾 Crop Recommendation (ML Prediction)

<table>
<tr>
<td align="center"><b>📝 Environmental Parameters</b></td>
<td align="center"><b>🎯 Predicted Optimal Crop</b></td>
</tr>
<tr>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/crop_input.jpg" width="400"/></td>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/crop_output.jpg" width="400"/></td>
</tr>
</table>

---

### 🧪 Fertilizer Suggestion (Data Analysis)

<table>
<tr>
<td align="center"><b>📊 Soil Nutrient Input</b></td>
<td align="center"><b>💡 Smart Recommendations</b></td>
</tr>
<tr>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/fertiliser_input.jpg" width="400"/></td>
<td><img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/fertiliser_output.jpg" width="400"/></td>
</tr>
</table>

---

### 📡 IoT Sensor Monitoring

<div align="center">
<img src="https://github.com/Shiva250503ss/Smart_Field/blob/main/output%20screenshots/LDR_Output.jpg" width="600"/>
<p><i>Real-time light intensity monitoring from IoT sensors</i></p>
</div>

</div>

---

## 🔗 API Architecture

```python
# RESTful Endpoints
┌────────────────────────────────────────────────────────────────┐
│  ENDPOINT              │  METHOD  │  DESCRIPTION               │
├────────────────────────┼──────────┼────────────────────────────┤
│  /                     │  GET     │  Homepage                  │
│  /crop-recommend       │  GET     │  Crop recommendation form  │
│  /crop-predict         │  POST    │  ML prediction endpoint    │
│  /disease-predict      │  GET/POST│  CNN inference endpoint    │
│  /fertilizer           │  GET     │  Fertilizer analysis form  │
│  /fertilizer-predict   │  POST    │  Nutrient recommendation   │
└────────────────────────┴──────────┴────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Package Manager)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Shiva250503ss/Smart_Field.git
cd Smart_Field/Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Navigate to `http://localhost:5000` in your browser.

---

## 📊 Skills Demonstrated

<div align="center">

| 🎯 Role | 🛠️ Relevant Skills from This Project |
|:-------:|:--------------------------------------|
| **Data Scientist** | Deep Learning (PyTorch), CNN Architecture Design, Random Forest, Feature Engineering, Model Training & Evaluation, Image Classification |
| **Data Analyst** | Data Processing (Pandas), Exploratory Analysis, CSV Data Management, Statistical Analysis, Data Visualization, Business Insights |
| **Data Engineer** | ETL Pipelines, REST API Development, Data Serialization (Pickle/JSON), Database Design, Production Deployment, Scalable Architecture |
| **AI/ML Engineer** | Computer Vision, PyTorch Implementation, Model Deployment, Real-time Inference, API Integration, End-to-End ML Systems |

</div>

---

## 🌟 Key Achievements

- 🎯 **38-class classification accuracy** for plant disease detection
- 🌐 **Real-time weather data integration** via OpenWeatherMap API
- 📊 **22+ crop types** supported in recommendation engine
- 🔧 **Production-ready deployment** with Gunicorn WSGI server
- 📡 **IoT-ready architecture** for sensor data integration
- 🎨 **Responsive web interface** with Bootstrap frontend

---

## 📚 Documentation

| Document | Description |
|:--------:|:------------|
| 📑 [Project Report](documents/FINAL%20YEAR%20PROJECT%20REPORT_SF.pdf) | Comprehensive technical documentation |
| 📑 [Research Paper](documents/Smart_Fields_Review_Paper_final1.pdf) | Academic review and methodology |

---

<div align="center">

## 📬 Connect With Me

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shiva250503ss)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shivaraj-senthil-rajan-2b8898227/)
[![Portfolio](https://img.shields.io/badge/Portfolio-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://shiva250503ss.github.io/shivaraj-portfolio/)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Shivaraj.SenthilRajan@colorado.edu)

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

**Made with 💚 for Sustainable Agriculture**

*If you found this project helpful, please consider giving it a ⭐!*

</div>
