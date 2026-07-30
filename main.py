import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('cat_dog_model.keras')

img = tf.keras.utils.load_img('test1.jpg', target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
print("Dog" if prediction > 0.5 else "Cat", f"({prediction:.2f})")
