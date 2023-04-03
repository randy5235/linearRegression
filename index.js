// Define the training data
const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // Input features
const output = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]; // Output labels

// Define the model parameters
let weight = 0; // Weight parameter of the model
let bias = 0; // Bias parameter of the model
const learningRate = 0.0001; // Learning rate for gradient descent
const numEpochs = 8000000000; // Number of epochs to train for
const earlyStoppingEpochs = 5;
let prevLoss = Infinity;
let numEpochsWithoutImprovement = 0;


// Define the loss function (squared error)
function meanSquaredError(prediction, label) {
    return (prediction - label) ** 2;
}

// function meanAbsoluteError(yPred, yTrue) {
//     const n = yTrue.length;
//     let errorSum = 0;
//     for (let i = 0; i < n; i++) {
//         errorSum += Math.abs(yTrue[i] - yPred[i]);
//     }
//     return errorSum / n;
// }

// Train the model using gradient descent
for (let epoch = 0; epoch < numEpochs; epoch++) {
    let lossSum = 0;
    for (let i = 0; i < input.length; i++) {
        const x = input[i]; // Get the input feature
        const yTrue = output[i]; // Get the output label

        // Forward pass: Make a prediction based on the current parameters
        const yPred = weight * x + bias;
        const l = meanSquaredError(yPred, yTrue); // Calculate the loss
        lossSum += l;

        // Backward pass: Calculate the gradients of the loss with respect to the parameters
        const dLdW = 2 * x * (yPred - yTrue);
        const dLdB = 2 * (yPred - yTrue);

        // Update the parameters based on the gradients and learning rate
        weight = weight - learningRate * dLdW;
        bias = bias - learningRate * dLdB;
    }

    // Print the average loss for the epoch
    const avgLoss = lossSum / input.length;
    console.log(`Epoch ${epoch + 1}: loss = ${avgLoss}, with ${weight} and ${bias}`);

    // Check if the loss has stopped improving
    if (avgLoss >= prevLoss) {
        numEpochsWithoutImprovement++;
    } else {
        numEpochsWithoutImprovement = 0;
    }
    prevLoss = avgLoss;

    // Stop training if the loss has stopped improving for too many epochs
    if (numEpochsWithoutImprovement >= earlyStoppingEpochs) {
        console.log(`Early stopping at epoch ${epoch + 1}`);
        break;
    }
}


// Use the trained model to predict the output of a new input
const guesses = [2, 3, 4, 5, 6, 7, 8, 9, 10, 23, 45, 75, 86];
guesses.forEach(guess => {

    const inputTest = guess;
    const outputPred = weight * inputTest + bias;
    console.log(`Prediction for ${inputTest} = ${outputPred}`);

});