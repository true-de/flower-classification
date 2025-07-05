import tensorflow as tf
import numpy as np

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Test without mixed precision
print("\nTest without mixed precision:")
predictions = tf.random.normal([10, 5])
targets = tf.random.uniform([10], maxval=5, dtype=tf.int32)
print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")

try:
    result = tf.math.in_top_k(targets=targets, predictions=predictions, k=2)
    print("Top-k operation successful without mixed precision")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error without mixed precision: {e}")

# Test with mixed precision
print("\nTest with mixed precision:")
tf.keras.mixed_precision.set_global_policy('mixed_float16')
predictions_fp16 = tf.random.normal([10, 5])
print(f"Predictions shape: {predictions_fp16.shape}, dtype: {predictions_fp16.dtype}")

try:
    result = tf.math.in_top_k(targets=targets, predictions=predictions_fp16, k=2)
    print("Top-k operation successful with mixed precision")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error with mixed precision: {e}")
    print("\nThis confirms the issue: mixed_float16 policy causes type mismatch with in_top_k operation")
    print("Solution: Either disable mixed precision or use float32 for this specific operation")

print("\nTest complete")