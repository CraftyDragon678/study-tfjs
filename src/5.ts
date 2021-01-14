import * as tf from '@tensorflow/tfjs-node';

export default async function run() {
  const temperature = [20, 21, 22, 23];
  const sale = [40, 42, 44, 46];

  const xdata = tf.tensor(temperature);
  const ydata = tf.tensor(sale);

  const X = tf.input({ shape: [1] });
  const Y = tf.layers.dense({ units: 1 }).apply(X) as tf.SymbolicTensor;
  const model = tf.model({ inputs: X, outputs: Y });
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  });

  const result = await model.fit(xdata, ydata, {
    epochs: 5000,
    callbacks: {
      onEpochEnd: (epoch, logs) => console.log('epoch', epoch, logs, 'RMSE=>', logs && Math.sqrt(logs.loss)),
    },
  });
  const predictY = model.predict(xdata) as tf.Tensor;
  predictY.print();

  const xpredict = tf.tensor([15, 16, 17, 18, 19]);

  (model.predict(xpredict) as tf.Tensor).print();

  await model.save('file://./lemon');
};
