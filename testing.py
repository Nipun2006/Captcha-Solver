from tensorflow.keras.models import load_model # type: ignore

# Load the saved model (this is fast and won't heat up your Mac!)
model = load_model('captcha_solver_v1.keras')

# Verify it's the right one
# model.summary()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # type: ignore

data = np.load('../Hopfield Network/character_font.npz')
images = data['images']
labels = data['labels']
X = images.reshape(-1,32,32,1)
X = X.astype('float32')/255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

def check_prediction(index):
    sample = X_test[index:index+1]
    actual = y_test[index]
    dist = model.predict(sample)[0]
    guess = np.argmax(dist)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: The Image
    ax1.imshow(sample.squeeze(), cmap='gray')
    ax1.set_title(f"Actual: {actual} | Guess: {guess}")
    ax1.axis('off')

    # Right: The Distribution
    ax2.bar(range(26), dist, color='salmon')
    ax2.set_xticks(range(26))
    ax2.set_title("Confidence Distribution")
    
    plt.tight_layout()
    plt.show()

# Run it on the first image
check_prediction(0)
# Get the distribution for the very first test image
distribution = model.predict(X_test[0:1])[0]

# Print it nicely
for i, prob in enumerate(distribution):
    print(f"Label {i}: {prob:.4f}")