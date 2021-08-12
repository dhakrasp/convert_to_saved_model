import tensorflow as tf

vocab_size = 512
hidden_size = 64
output_classes = 4


class SiameseModel(tf.keras.models.Model):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True)
        self.lstm_layer = tf.keras.layers.LSTM(hidden_size)
        self.dense_layer = tf.keras.layers.Dense(output_classes, activation='softmax')

    # Decorate every function that needs to be exposed
    @tf.function
    def call(self, inputs, training=False):
        text_ids_1 = inputs[0]
        text_ids_2 = inputs[1]
        embedded_1 = self.embedding_layer(text_ids_1)
        embedded_2 = self.embedding_layer(text_ids_2)
        hidden_1 = self.lstm_layer(embedded_1)
        hidden_2 = self.lstm_layer(embedded_2)
        final_rep = tf.concat([hidden_1, hidden_2, hidden_1 - hidden_2], axis=-1)
        return self.dense_layer(final_rep)

    # Decorate every function that needs to be exposed
    @tf.function
    def get_hidden_rep(self, text_ids):
        embedded = self.embedding_layer(text_ids)
        return self.lstm_layer(embedded)


if __name__ == '__main__':
    inputs = [
        tf.zeros((5, 7)),
        tf.ones((5, 7))
    ]
    model = SiameseModel()
    outputs = model.call(inputs)
    print(outputs.shape)
    hidden_rep = model.get_hidden_rep(inputs[0])
    print(hidden_rep.shape)
