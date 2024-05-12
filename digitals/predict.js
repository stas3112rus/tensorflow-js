import {
	TRAINING_DATA
} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';

const context = document.querySelector('canvas').getContext('2d');
const prediction = document.querySelector('#prediction');
const info = document.querySelector('#info');
const progress = document.querySelector('progress');

// Рисование изображения на основе числовых данных
const drawImage = (digit) => {
	let imageData = context.getImageData(0, 0, 28, 28);
	for (let i = 0; i < digit.length; i++) {
		imageData.data[i * 4] = digit[i] * 255; // red
		imageData.data[i * 4 + 1] = digit[i] * 255; // green
		imageData.data[i * 4 + 2] = digit[i] * 255; // blue
		imageData.data[i * 4 + 3] = 255;
	}
	context.putImageData(imageData, 0, 0);
};

// Создание стартовых тензоров
const INPUT_TENSOR = tf.tensor2d(TRAINING_DATA.inputs);
const OUTPUT_TENSOR = tf.oneHot(tf.tensor1d(TRAINING_DATA.outputs, 'int32'), 10);

// Создание модели
let model = tf.sequential();
model.add(tf.layers.dense({
	inputShape: [784],
	units: 32
}));
model.add(tf.layers.dense({
	units: 16
}));
model.add(tf.layers.dense({
	units: 10,
	activation: 'softmax'
}));

// Обучение модели
const train = async () => {
	model.compile({
		optimizer: tf.train.adam(0.01),
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy'],
	});
	info.innerText = 'Training model. Please wait...';
	progress.style.display = 'block';
	const epochs = 50;
	await model.fit(INPUT_TENSOR, OUTPUT_TENSOR, {
		batchSize: 512,
		epochs,
		shuffle: true,
		callbacks: {
			onEpochEnd: (epoch, logs) => {
				progress.value = epoch / (epochs - 1) * 100;
				console.log(`Epoch ${epoch} `, logs);
			}
		},
	});
	info.innerText = 'Model succesfully trained';
	progress.style.display = 'none';
	// Удаление стартовых тензоров
	INPUT_TENSOR.dispose();
	OUTPUT_TENSOR.dispose();
}
await train();

const tryToPredict = () => {
	let randomIndex = Math.floor(Math.random() * TRAINING_DATA.inputs.length);
	let result = tf.tidy(() => {
		let newInput = tf.tensor1d(TRAINING_DATA.inputs[randomIndex]);
		let output = model.predict(newInput.expandDims());
		return output.squeeze().argMax();
	});
	result.array().then(number => {
		prediction.innerText = number;
		prediction.style.color = (number === TRAINING_DATA.outputs[randomIndex]) ?
			'#00ff00' // green
			:
			'#ff0000'; // red
		drawImage(TRAINING_DATA.inputs[randomIndex]);
	});
};
setInterval(() => tryToPredict(), 1000);