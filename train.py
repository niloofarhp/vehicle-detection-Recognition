import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time

from tensorflow.python.keras.layers.core import Activation


def SimlpleModel():
	model = keras.Sequential([
	layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),
	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(2)
])
	# model = keras.Sequential()
	# model.add(keras.Input(shape=(96, 96, 3)))
	# model.add(layers.Conv2D(32, 5, strides=2, padding="same", activation="relu"))
	# model.add(layers.MaxPooling2D(2))
	# model.add(layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"))
	# model.add(layers.MaxPooling2D(2))
	# model.add(layers.Conv2D(16, 3, strides=3, padding="same", activation="relu"))
	# model.add(layers.MaxPooling2D(2))
	# model.add(layers.Flatten())
	# model.add(layers.Dense(100))
	# model.add(layers.Dense(100))
	# model.add(layers.Dense(2, activation = "softmax"))

	model.summary()
	return model


def train(model, train_ds, val_ds):	
	print("start training")
	optimizer = keras.optimizers.SGD(learning_rate=1e-3)
	loss_fn = keras.losses.SparseCategoricalCrossentropy() #keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy() #keras.metrics.SparseCategoricalAccuracy()
	val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy() #keras.metrics.SparseCategoricalAccuracy()
	epochs = 10
	batch_size = 5
	for epoch in range(epochs):
		print("\nStart of epoch %d" % (epoch,))
		start_time = time.time()
		for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
			with tf.GradientTape() as tape:
				logits = model(x_batch_train, training=True)
				#y_pred_cls = tf.argmax(tf.squeeze(logits), axis=1)
				#y_pred_cls = tf.cast(y_pred_cls, tf.float32)
				loss_value = loss_fn(y_batch_train, logits)
			grads = tape.gradient(loss_value, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			train_acc_metric.update_state(y_batch_train, logits)
			if step % 200 == 0:
				print(
					"Training loss (for one batch) at step %d: %.4f"
					% (step, float(loss_value))
				)
				print("Seen so far: %d samples" % ((step + 1) * batch_size))
		train_acc = train_acc_metric.result()
		print("Training acc over epoch: %.4f" % (float(train_acc),))
		train_acc_metric.reset_states()

		for x_batch_val, y_batch_val in val_ds:
			val_logits = model(x_batch_val, training=False)
			val_acc_metric.update_state(y_batch_val, val_logits)
		val_acc = val_acc_metric.result()
		val_acc_metric.reset_states()
		print("Validation acc: %.4f" % (float(val_acc),))
		print("Time taken: %.2fs" % (time.time() - start_time))


def main():
	print("load dataset train/ test")
	img_height =180 #200
	img_width = 180 #200
	batch_s = 5
	data_dir = "D:/LPR_Projects/vehicle-detection-recognition-Data/train"
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_s)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_s)

	class_names = train_ds.class_names
	print(class_names)
	train(SimlpleModel(), train_ds, val_ds)

if __name__ == "__main__":
    main()	