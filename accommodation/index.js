import tf, {
	tensor
} from '@tensorflow/tfjs';
import {
	TRAINING_DATA
} from './data.js';

console.log("count of TRAINING_DATA.inputs.length:", TRAINING_DATA.inputs.length)

//Данные
const inputTensor = tf.tensor2d(TRAINING_DATA.inputs);
const outputTensor = tf.tensor1d(TRAINING_DATA.outputs);

//Нормализация данных
const minInputValue = tf.min(inputTensor, 0);
const maxInputValue = tf.max(inputTensor, 0);
const rangeSize = tf.sub(maxInputValue, minInputValue);

const normalize = (tensor) => {
	const result = tf.tidy(() => {
		const subtractTensor = tf.sub(tensor, minInputValue);
		return tf.div(subtractTensor, rangeSize);
	})

	return result;
}

const normalizedInputTensor = normalize(inputTensor);

//Создание модели
const model = tf.sequential();
model.add(tf.layers.dense({
	inputShape: [2], // нейронов на входе
	units: 1 // нейронов на выходе
}))

//Обучение модели 

const train = async () => {
	// Компиляция модели
	// Оптимизация и функция потерь
	model.compile({
		optimizer: tf.train.sgd(0.01), // метод Градиентные стохастический спуск, learning rate = 0.01
		loss: 'meanSquaredError' // функция потерь - Среднеквадратическая ошибка
	})

	//Обучение
	await model.fit(normalizedInputTensor, outputTensor, {
		batchSize: 64, // размер пакета
		epochs: 10,
		shuffle: true
	});

	//удаляем не нужные тензоры из памяти
	normalizedInputTensor.dispose();
	outputTensor.dispose();
}

await train();

// Тестирование 

const tryPredict = (dataForPredict) => {
	tf.tidy(() => {
		// Нормализованный тензор входящих данных
		const normalizedData = normalize(tf.tensor2d(dataForPredict))
		const output = model.predict(normalizedData);
		console.log('Предполагаемая цена на дома', dataForPredict);
		output.print();
	})
}

tryPredict([[3056, 3], [1034, 2], [950, 1]])

// Удаление оставшихся тензоров из памяти 
minInputValue.dispose();
maxInputValue.dispose();
outputTensor.dispose();