import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

class ResBlock(tf.keras.layers.Layer):
  def __init__(self, kernel_size, filters):
    super(ResBlock, self).__init__(name='residualblock')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (3, 3))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (3, 3))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=True):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

class EpiNN(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(EpiNN, self).__init__(obs_space,action_space,num_outputs,model_config,name)
        self.inp_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        self.res0 = ResBlock(kernel_size=(3, 3), filters=[64, 64, 64])(self.inp_layer)
        self.res1 = ResBlock(kernel_size=(3, 3), filters=[64, 64, 64])(self.res0)
        self.res2 = ResBlock(kernel_size=(3, 3), filters=[64, 64, 64])(self.res1)
        self.res3 = ResBlock(kernel_size=(3, 3), filters=[64, 64, 64])(self.res2)
        self.resP = ResBlock(kernel_size=(3, 3), filters=[64, 64, 64])(self.res3)
        self.resV = ResBlock(kernel_size=(3, 3), filters=[64, 64, 64])(self.res3)
        self.Flatten0 = tf.keras.layers.Flatten()(self.resP)
        self.Flatten1 = tf.keras.layers.Flatten()(self.resV)
        self.actor = tf.keras.layers.Dense(1, activation='tanh')(self.Flatten1)
        self.critic = tf.keras.layers.Dense(100, activation='tanh')(self.Flatten0)
        self.base_model = tf.keras.Model(self.inp_layer, [self.actor, self.critic])
    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state







