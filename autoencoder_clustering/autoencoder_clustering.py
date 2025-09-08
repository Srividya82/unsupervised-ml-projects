import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Autoencoder
encoding_dim = 64
input_img = layers.Input(shape=(784,))
encoded = layers.Dense(encoding_dim, activation="relu")(input_img)
decoded = layers.Dense(784, activation="sigmoid")(encoded)
autoencoder = models.Model(input_img, decoded)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# Extract encoded features
encoder = models.Model(input_img, encoded)
features = encoder.predict(x_test)

# Apply KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(features)

# Show sample from each cluster
for i in range(10):
    idx = (labels == i)
    plt.imshow(x_test[idx][0].reshape(28, 28), cmap="gray")
    plt.title(f"Cluster {i}")
    plt.show()
