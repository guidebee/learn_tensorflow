
�root"_tf_keras_sequential*�{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAUAAAADAAAA80QAAACXAHQAAAAAAAAAAAAAAKABAAAAAAAAAAAAAAAAAAAA\nAAAAAAB8AHwAZAF6CAAAZwJkAqwDpgIAAKsCAAAAAAAAAABTACkETukCAAAA6QEAAAApAdoEYXhp\ncykC2gJ0ZtoFc3RhY2spAdoBeHMBAAAAIPofPGlweXRob24taW5wdXQtMzAtNzE1Y2JmYWE3ZWZk\nPvoIPGxhbWJkYT5yCQAAAAIAAABzHgAAAIAApVKnWKJYqHGwIbBRsSSoabhhoFjRJUDUJUCAAPMA\nAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": "random_normal", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 4, "build_input_shape": {"class_name": "TensorShape", "items": [null]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null]}, "float32", "lambda_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null]}, "float32", "lambda_input"]}, "keras_version": "2.14.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}, "shared_object_id": 0}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAUAAAADAAAA80QAAACXAHQAAAAAAAAAAAAAAKABAAAAAAAAAAAAAAAAAAAA\nAAAAAAB8AHwAZAF6CAAAZwJkAqwDpgIAAKsCAAAAAAAAAABTACkETukCAAAA6QEAAAApAdoEYXhp\ncykC2gJ0ZtoFc3RhY2spAdoBeHMBAAAAIPofPGlweXRob24taW5wdXQtMzAtNzE1Y2JmYWE3ZWZk\nPvoIPGxhbWJkYT5yCQAAAAIAAABzHgAAAIAApVKnWKJYqHGwIbBRsSSoabhhoFjRJUDUJUCAAPMA\nAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": "random_normal", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>SGD", "config": {"name": "SGD", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": false, "is_legacy_optimizer": false, "learning_rate": 0.009999999776482582, "momentum": 0.0, "nesterov": false}}}}2
�root.layer-0"_tf_keras_layer*�{"name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAUAAAADAAAA80QAAACXAHQAAAAAAAAAAAAAAKABAAAAAAAAAAAAAAAAAAAA\nAAAAAAB8AHwAZAF6CAAAZwJkAqwDpgIAAKsCAAAAAAAAAABTACkETukCAAAA6QEAAAApAdoEYXhp\ncykC2gJ0ZtoFc3RhY2spAdoBeHMBAAAAIPofPGlweXRob24taW5wdXQtMzAtNzE1Y2JmYWE3ZWZk\nPvoIPGxhbWJkYT5yCQAAAAIAAABzHgAAAIAApVKnWKJYqHGwIbBRsSSoabhhoFjRJUDUJUCAAPMA\nAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": "random_normal", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}2
�:root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 6}2