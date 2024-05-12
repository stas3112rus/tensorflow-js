const image = document.querySelector('img');
const context = document.querySelector('canvas').getContext('2d');

//используем обученную функцию

const loadAndRunModel = async () => {
	const model = await tf.loadGraphModel(
		'http://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4', {
			fromTFHub: true
		}
	);

	// Подготовка данных 
	const imageTensor = tf.browser.fromPixels(image);
	const resizedTensor = tf.image.resizeBilinear(imageTensor, [192, 192], true).toInt();

	// Использование модели
	const tensorOutput = model.predict(tf.expandDims(resizedTensor));


	// Отображение результата
	tensorOutput.array().then(array => {
		console.log(array)

		array[0][0].map(elem => {
			const x = elem[1] * image.width;
			const y = elem[0] * image.height;
			context.fillStyle = '#04ff00';
			context.fillRect(x - 5, y - 5, 10, 10);
		})
	});
}

await loadAndRunModel();