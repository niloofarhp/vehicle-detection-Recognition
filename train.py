import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
def model():
	model = keras.Sequential()
	model.add(keras.Input(shape=(96, 96, 3)))
	model.add(layers.Conv2D(32, 5, strides=2, padding="same", activation="relu"))
	model.add(layers.MaxPooling2D(2))
	model.add(layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"))
	model.add(layers.MaxPooling2D(2))
	model.add(layers.MaxPooling2D(100))
	model.add(layers.MaxPooling2D(100))
	model.add(layers.Dense(2))
	model.summary()
def train():	
	print("start training")
	# Instantiate an optimizer.
	optimizer = keras.optimizers.SGD(learning_rate=1e-3)
	# Instantiate a loss function.
	loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
	val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
	epochs = 2
	batch_size = 5
	for epoch in range(epochs):
		print("\nStart of epoch %d" % (epoch,))
		start_time = time.time()

		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			with tf.GradientTape() as tape:
				logits = model(x_batch_train, training=True)
				loss_value = loss_fn(y_batch_train, logits)
			grads = tape.gradient(loss_value, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			# Update training metric.
			train_acc_metric.update_state(y_batch_train, logits)

			# Log every 200 batches.
			if step % 200 == 0:
				print(
					"Training loss (for one batch) at step %d: %.4f"
					% (step, float(loss_value))
				)
				print("Seen so far: %d samples" % ((step + 1) * batch_size))

		# Display metrics at the end of each epoch.
		train_acc = train_acc_metric.result()
		print("Training acc over epoch: %.4f" % (float(train_acc),))

		# Reset training metrics at the end of each epoch
		train_acc_metric.reset_states()

		# Run a validation loop at the end of each epoch.
		for x_batch_val, y_batch_val in val_dataset:
			val_logits = model(x_batch_val, training=False)
			# Update val metrics
			val_acc_metric.update_state(y_batch_val, val_logits)
		val_acc = val_acc_metric.result()
		val_acc_metric.reset_states()
		print("Validation acc: %.4f" % (float(val_acc),))
		print("Time taken: %.2fs" % (time.time() - start_time)
def main():
	print("load dataset train/ test")
	
if __name__ == "__main__":
    main()	