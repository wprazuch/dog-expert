import tensorflow as tf


class Preprocessor:

    def __init__(self):
        self.operations = []

    def cast(self, dtype):
        def cast_operation(data):
            for k, v in data.items():
                data[k] = tf.cast(v, dtype)
            return data

        self.operations.append(cast_operation)

        return self

    def normalize(self):
        def normalize_operation(data):
            for k, v in data.items():
                data[k] = v / 255.0
            return data

        self.operations.append(normalize_operation)
        return self

    def add_to_graph(self, dataset) -> tf.data.Dataset:

        for operation in self.operations:
            dataset = dataset.map(operation)

        return dataset
