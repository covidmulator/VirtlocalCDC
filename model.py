import tensorflow as tf

class ResBlock(tf.keras.layers.Layer):
  def __init__(self, kernel_size, filters):
    super(ResBlock, self).__init__(name='')
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

class EpiNN(tf.keras.Model):
    def __init__(self,inp,learning_rate):
        super(EpiNN,self).__init__()
        self.inp = inp
        self.lr = learning_rate
        self.inp_layer = tf.keras.layers.Input(inp)
        self.res0 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res1 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res2 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res3 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res4 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res5 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res6 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res7 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res8 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res9 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res10 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res11 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res13 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res14 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res15 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.res16 = ResBlock(kernel_size=(3, 3), filters=[64,64,64])
        self.prob = tf.keras.layers.Dense(activation='linear')
        self.output = tf.keras.layers.Dense(units=1, activation='tanh')
    def build_model(self,input):
        x = self.inp_layer()(input)
        x = self.res0()(x)
        x = self.res1()(x)
        x = self.res2()(x)
        x = self.res3()(x)
        x = self.res4()(x)
        x = self.res5()(x)
        x = self.res6()(x)
        x = self.res7()(x)
        x = self.res8()(x)
        x = self.res9()(x)
        x = self.res10()(x)
        x = self.res11()(x)
        x = self.res12()(x)
        x = self.res13()(x)
        x = self.res14()(x)
        x1 = self.res15()(x)
        x2 = self.res16()(x)
        FlattenP = tf.keras.layers.Flatten()(x1)
        DenseP = tf.keras.layers.Dense(self.inp**2)(FlattenP)
        Policy = self.prob(DenseP)
        FlattenV = tf.keras.layers.Flatten()(x2)
        DenseV = tf.keras.layers.Dense(10)(FlattenV)
        Value = self.output(DenseV)
        model = tf.keras.Model(input=self.inp_layer, outputs=[Value, Policy])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': tf.nn.softmax_cross_entropy_with_logits},
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})
        return model








