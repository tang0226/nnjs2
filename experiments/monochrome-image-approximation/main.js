/**

Rendering / training pipeline:

When the training session is triggered, the main script sends hyperparameters
to the master worker, which 


main script handles creation / termination of workers, as well as
sending hyperparameter updates to the workers.


Render worker(s) (responsible for a chunk of the rendered image)
Learning worker(s)


Tasks:
1. Run some number of epochs, noting the parameter deltas
2. Apply the parameter deltas and draw the image (using requestAnimationFrame with putImageData)
3. repeat


Batching optiosn:
1. Full-batch (run through every point in the training dataset, then end the epoch and apply the average derivative)
2. Stochastic  / mini-batch: more frequent updates using random sampling and smaller batch sizes

*/

function rgbToGrayscale(r, g, b) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}


function enableInput(ele)  {
  ele.removeAttribute("disabled");
}

function disableInput(ele) {
  ele.setAttribute("disabled", "true");
}


const mainCanvas = document.getElementById("main-canvas");
const ctx = mainCanvas.getContext("2d");

const targetCanvas = document.getElementById("target-canvas");
const targetCtx = targetCanvas.getContext("2d");

const imageInput = document.getElementById("image-input");
const canvasSizeInput = document.getElementById("canvas-size");

const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const resetNetworkButton = document.getElementById("reset-network-button");

const trainingWorkersInput = document.getElementById("training-workers");
const renderWorkersInput = document.getElementById("render-workers");

const activationFunctionInput = document.getElementById("activation-function");
const hiddenLayersInput = document.getElementById("hidden-layers");
const learningRateInput = document.getElementById("learning-rate");
const epfInput = document.getElementById("epf-input");

// Hyperparameters that cannot be changed during training
const coreInputs = [
  trainingWorkersInput,
  renderWorkersInput,
  activationFunctionInput,
  hiddenLayersInput
];


canvasSizeInput.value = "200";

hiddenLayersInput.value = "40, 20";
learningRateInput.value = "0.2";
epfInput.value = "1";

var width, height;

function setCanvasDim(w, h) {
  mainCanvas.width = w;
  mainCanvas.height = h;
  mainCanvas.style.width = w + "px";
  mainCanvas.style.height = h + "px";
  targetCanvas.width = w;
  targetCanvas.height = h;
  targetCanvas.style.width = w + "px";
  targetCanvas.style.height = h + "px";
  width = w;
  height = h;
}

var canvasSize = Number(canvasSizeInput.value);
setCanvasDim(canvasSize, canvasSize);

// Stores the size of the raw input image
var imgFileWidth = imgFileHeight = 1000;
var targetImgData;

// Update canvas dimensions based on the target canvas size and the AR of the image file
function updateCanvasDim() {
  let ar = imgFileWidth / imgFileHeight;
  if (ar > 1) {
    setCanvasDim(Math.round(canvasSize / ar), canvasSize);
  }
  else {
    setCanvasDim(canvasSize, Math.round(canvasSize / ar));
  }
}


// Scale of the neural network's inputs vs. canvas
var xScale = 2, yScale = 2, xCenter = 0, yCenter = 0;

function planeXToCanvas(x) {
  return width / 2 + width * (x - xCenter) / xScale;
}

function planeYToCanvas(y) {
  return height / 2 - height * (y - yCenter) / yScale;
}

function canvasXToPlane(x) {
  return xCenter + xScale * (x - width / 2) / width;
}

function canvasYToPlane(y) {
  return yCenter - yScale * (y - height / 2) / height;
}


function formatHiddenLayers(layers) {
  return layers.join(", ");
}

function parseHiddenLayersString(str) {
  let nums = str.split(",").map((x) => Number(x.trim()));
  for (let n of nums) {
    if (Number.isNaN(n) || !Number.isInteger(n)) {
      return false;
    }
  }
  return nums;
}


function updateActivationFunction() {
  let val = activationFunctionInput.value;
  switch(val) {
    case "relu":
      activationFunction = NN.RELU;
      break;
    case "sigmoid":
      activationFunction = NN.SIGMOID;
      break;
    case "tanh":
      activationFunction = NN.TANH;
      break;
  }
}


var hiddenLayers = parseHiddenLayersString(hiddenLayersInput.value);
var activationFunction;
updateActivationFunction();

var agent = {
  isTraining: false,
  learningRate: Number(learningRateInput.value),

  initNetwork(hiddenLayers, af) {
    this.nn = new NN({
      layerSizes: [2].concat(hiddenLayers).concat([3]),
      activationFunctions: [af, NN.SIGMOID],
      wInit: {
        method: NN.RANDOM,
        range: 0.5,
      },
      bInit: {
        method: NN.RANDOM,
        range: 0.5,
      },
    });
  },

  draw() {

  },

  learn() {

  }
}
agent.initNetwork(hiddenLayers, activationFunction);

var epoch = 0;
var epochsPerFrame = Number(epfInput.value);

stopButton.setAttribute("disabled", true);

function draw() {

}


imageInput.addEventListener("change", (event) => {
  var fr = new FileReader;

  fr.onload = function() {
    var img = new Image();
    img.onload = function() {
      imgFileWidth = img.width;
      imgFileHeight = img.height;
      updateCanvasDim();

      console.log(img);
      //mainImg = img;

      //imgFileWidth = mainImg.width;
      //imgFileHeight = mainImg.height;

      //updateCanvasSizes(canvasSize, imgFileWidth, imgFileHeight);
      targetCtx.drawImage(img, 0, 0, width, height);
      //mainImgData = ctx.getImageData(0, 0, width, height).data;
    };

    img.src = fr.result;
  }

  if (event.target.files.length) {
    fr.readAsDataURL(event.target.files[0]);
  }
});

canvasSizeInput.addEventListener("change", () => {
  let n = Number(canvasSizeInput.value);
  if (Number.isNaN(n) || !Number.isInteger(n) || n <= 0) {
    canvasSizeInput.value = canvasSize;
  }
  else {
    canvasSize = n;
    updateCanvasDim();
  }
});

hiddenLayersInput.addEventListener("change", () => {
  let layers = parseHiddenLayersString(hiddenLayersInput.value);
  if (layers) {
    // update the layers and reset the agent
    hiddenLayers = layers;
    agent.initNetwork(hiddenLayers, activationFunction);
    epoch = 0;
  }
  else {
    hiddenLayersInput.value = formatHiddenLayers(hiddenLayers);
  }
});

activationFunctionInput.addEventListener("change", () => {
  // Update the af and reset the agent
  updateActivationFunction();
  agent.initNetwork(hiddenLayers, activationFunction);
  epoch = 0;
});

learningRateInput.addEventListener("change", () => {
  let n = Number(learningRateInput.value);
  if (Number.isNaN(n)) {
    learningRateInput.value = agent.learningRate;
  }
  else {
    agent.learningRate = n;
  }
});

epfInput.addEventListener("change", () => {
  let n = Number(epfInput.value);
  if (Number.isNaN(n) || !Number.isInteger(n)) {
    epfInput.value = epochsPerFrame;
  }
  else {
    epochsPerFrame = n;
  }
});

startButton.addEventListener("click", () => {
  agent.isTraining = true;
  disableInput(startButton);
  enableInput(stopButton);
  coreInputs.forEach((input) => {disableInput(input)});
});

stopButton.addEventListener("click", () => {
  agent.isTraining = false;
  disableInput(stopButton);
  enableInput(startButton);
  coreInputs.forEach((input) => {enableInput(input)});
});

resetNetworkButton.addEventListener("click", () => {

});

var keys = {};

window.addEventListener("keydown", (e) => {
  keys[e.key] = true;
});

window.addEventListener("keyup", (e) => {
  keys[e.key] = false;
});
