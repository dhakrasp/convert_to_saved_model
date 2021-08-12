import tensorflow as tf
from siamese_model import SiameseModel

inputs = [
    tf.zeros((5, 7)),
    tf.ones((5, 7))
]

model = SiameseModel()

# Call/Fit/Predict model once after loading. This builds all the layers
outputs = model(inputs)
outputs = model.call(inputs)
outputs = model.predict(inputs)
print(outputs.shape)
hidden_rep = model.get_hidden_rep(inputs[0])
print(hidden_rep.shape)

print(model.summary())

# Input shape should be None, None for variable-length inputs
# Input shape should be (say) None, 25 for fixed-length (25) inputs
call_input_signature = [
    tf.TensorSpec([None, None], dtype=tf.float32),
    tf.TensorSpec([None, None], dtype=tf.float32),
]

get_hidden_rep_input_signature = tf.TensorSpec([None, None], dtype=tf.float32)

call_concrete_func = model.call.get_concrete_function(call_input_signature)
get_hidden_rep_concrete_func = model.get_hidden_rep.get_concrete_function(get_hidden_rep_input_signature)

# Typically call() will be the default service rendered
signatures = {
    'serving_default': call_concrete_func,
    'get_hidden_rep': get_hidden_rep_concrete_func
}

saved_model_dir = 'saved_model/'

model.save(saved_model_dir, save_format='tf', signatures=signatures)

saved_model = tf.saved_model.load(saved_model_dir)
print(list(saved_model.signatures.keys()))  # ["serving_default"]

outputs = saved_model.call(inputs)
print(outputs.shape)
hidden_rep = saved_model.get_hidden_rep(inputs[0])
print(hidden_rep.shape)
