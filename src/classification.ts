import * as dfd from 'danfojs-node';
import * as tf from '@tensorflow/tfjs-node';

export default async function run() {
  const data: dfd.DataFrame = await dfd.read_csv('https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv');
  data.print();
  const xdata = data.loc({ columns: ['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭'] });
  xdata.print();
  const encoder = new dfd.OneHotEncoder();
  const ydata = encoder.fit((<any>data)['품종']);
  ydata.print();
  console.log(data.shape);

  const X = tf.input({ shape: [4] });
  const H = tf.layers.dense({ units: 4, activation: 'relu' }).apply(X);
  const Y = tf.layers.dense({ units: 3, activation: 'softmax' }).apply(H);

  const model = tf.model({ inputs: X, outputs: Y as tf.SymbolicTensor });
  model.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await model.fit(
    xdata.tensor as unknown as tf.Tensor2D,
    ydata.tensor as unknown as tf.Tensor2D,
    {
      epochs: 200,
      callbacks: {
        onEpochEnd: (epoch, logs) => console.log(logs?.acc),
      },
    },
  );
}
