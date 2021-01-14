import * as tf from '@tensorflow/tfjs-node';
import { xdata, ydata } from './10data';

export default async function run() {
  const x = tf.tensor(xdata);
  const y = tf.tensor(ydata);

  const X = tf.input({ shape: [13] });
  const Y = tf.layers.dense({ units: 1 }).apply(X) as tf.SymbolicTensor;

  const model = tf.model({ inputs: X, outputs: Y });
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  });

  await model.fit(x, y, {
    epochs: 5000,
  });
  const weights = model.getWeights();
  console.log(await weights[0].array());
  console.log(await weights[1].array());
}

