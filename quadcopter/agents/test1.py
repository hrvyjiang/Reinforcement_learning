import tensorflow as tf

print("TF version:", tf.__version__)
print("Number GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Number CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))