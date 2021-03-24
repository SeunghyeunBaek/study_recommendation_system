
import tensorflow as tf

class BatchTrainer():
    
    def __init__(self, model, optimizer, metric, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.metric = metric
        self.loss_function = loss_function

    def train(self, x, y):
        
        with tf.GradientTape() as tape:

            y_pred = self.model(x)
            loss = self.loss_function(y_true=y, y_pred=y_pred)
        
        grad = tape.gradient(target=loss, sources=self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        self.metric.update_state(y, y_pred)

        return loss.numpy(), self.metric.result().numpy()

    def validate(self, x, y):

        y_pred  = self.model(x)
        loss = self.loss_function(y_true=y, y_pred=y_pred)
        self.metric.update_state(y, y_pred)

        return loss.numpy(), self.metric.result().numpy() 