import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image
img = cv2.imread("sample.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape pixels
pixels = img.reshape((-1, 3))

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(pixels)
segmented_img = kmeans.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)

# Display
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title("Segmented")

plt.show()
