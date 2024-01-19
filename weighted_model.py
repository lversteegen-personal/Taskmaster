import keras
import tensorflow as tf


@tf.function
def weighted_reward_loss(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor = None):

    log = tf.math.multiply(tf.abs(tf.math.log(y_pred+1e-5)), y_true)
    square = tf.square(y_pred-y_true)

    if weights == None:
        return tf.reduce_mean(square*(1+log), axis=-1)
    else:
        return tf.reduce_sum(square*(1+log)*weights, axis=-1)

@tf.function
def weighted_confidence_loss(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor = None):

    log = tf.maximum(0.0, tf.math.log(y_true+1e-5)-tf.math.log(y_pred+1e-5))
    square = tf.square(y_pred-y_true)

    if weights == None:
        return tf.reduce_mean(log+square, axis=-1)
    else:
        return tf.reduce_sum((log+square)*weights, axis=-1)

@tf.function
def weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor = None):

    square = tf.square(y_pred-y_true)

    if weights == None:
        return tf.reduce_mean(square, axis=-1)
    else:
        return tf.reduce_sum(square*weights, axis=-1)


class WeightedModel(keras.Model):

    def __init__(self, *args, **kwargs):
        tf.random.set_seed(0)
        super(WeightedModel, self).__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean("loss")
        self.value_loss_tracker = keras.metrics.Mean("value_loss")
        self.reward_loss_tracker = keras.metrics.Mean("reward_loss")
        self.confidence_loss_tracker = keras.metrics.Mean("confidence_loss")

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.value_loss_tracker.reset_states()
        self.reward_loss_tracker.reset_states()
        self.confidence_loss_tracker.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker, self.value_loss_tracker, self.reward_loss_tracker, self.confidence_loss_tracker]

    #@tf.function
    def train_step(self: "WeightedModel", data):

        x, y_true, weights = data

        # compute loss
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            value_loss = weighted_mse(y_true[0][:,None], y_pred[0])
            reward_loss = weighted_mse(y_true[1], y_pred[1], weights)
            reward_confidence_loss = weighted_confidence_loss(
                y_true[2], y_pred[2], weights)
            loss = value_loss + reward_loss + reward_confidence_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.value_loss_tracker.update_state(value_loss)
        self.reward_loss_tracker.update_state(reward_loss)
        self.confidence_loss_tracker.update_state(reward_confidence_loss)

        return {m.name: m.result() for m in self.metrics}
