# 🧠 Unsupervised Learning & Segmentation Projects  

This repository contains a collection of projects implemented in **Python** and tested on **Google Colab**.  
The focus is on **Unsupervised Learning** (clustering) and **Image Segmentation** techniques using **ML & AI**.  

---

## 📂 Projects  

### 1. Customer Segmentation (K-Means Clustering)  
- **Goal**: Group customers based on annual income and spending score.  
- **Dataset**: [Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)  
- **File**: `customer_segmentation/customer_segmentation.py`  

### 2. Image Segmentation (K-Means on Pixels)  
- **Goal**: Segment an image into regions based on pixel color clusters.  
- **File**: `image_segmentation/image_segmentation.py`  

### 3. Autoencoder + KMeans (Image Clustering on MNIST)  
- **Goal**: Learn compressed features of MNIST digits using an autoencoder, then cluster digits without labels.  
- **File**: `autoencoder_clustering/autoencoder_clustering.py`  

---

## 🚀 How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/unsupervised-ml-projects.git
   cd unsupervised-ml-projects
   ```

2. Install requirements:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run projects:  
   ```bash
   python customer_segmentation/customer_segmentation.py
   python image_segmentation/image_segmentation.py
   python autoencoder_clustering/autoencoder_clustering.py
   ```

---

## 📦 Requirements  
- Python 3.8+  
- scikit-learn  
- numpy  
- matplotlib  
- pandas  
- opencv-python  
- tensorflow  
