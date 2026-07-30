import tensorflow as tf

# Load Dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    'Dataset',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

val_data = tf.keras.utils.image_dataset_from_directory(
    'Dataset',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

#CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=100)

model.save('cat_dog_model.keras')

print("Model trained and saved as cat_dog_model.keras")

