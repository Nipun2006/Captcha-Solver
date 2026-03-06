import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

data = np.load('../Hopfield Network/character_font.npz')
images = data['images']
labels = data['labels']

X = images.reshape(-1,32,32,1)
X = X.astype('float32')/255.0
"""
plt.figure(figsize=(15,3))
plt.suptitle('Character Font Dataset', fontsize=16)
for i in range(26):
    plt.subplot(2,13,i+1)
    plt.imshow(X[i].reshape(32,32), cmap='gray')
    plt.axis('off')
plt.show()
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training on: {X_train.shape[0]} images")
print(f"Testing on: {X_test.shape[0]} images")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))

# Since you have 26 letters (A-Z), the final layer needs 26 outputs
model.add(layers.Dense(26, activation='softmax'))

# Compile tells the model how to learn
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# This starts the actual training process
history = model.fit(X_train, y_train, epochs=5, 
                    validation_data=(X_test, y_test), 
                    batch_size=64)

# Save the entire model to a single file
model.save('captcha_solver_v1.keras') 
print("Model saved! Your laptop can rest now.")

# This gives you the overall accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")