const mainCanvas = document.getElementById("main-canvas");
const ctx = mainCanvas.getContext("2d");

const evolvingCanvas = document.getElementById("evolving-canvas");
const eCtx = evolvingCanvas.getContext("2d");

const canvasSizeInput = document.getElementById("canvas-size-input");
const imageFileInput = document.getElementById("image-file-input");
const learningRateInput = document.getElementById("learning-rate-input");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const restartButton = document.getElementById("restart-button");

stopButton.setAttribute("disabled", true);

var width, height;
function setCanvasDim(w, h) {
  mainCanvas.width = w;
  mainCanvas.height = h;
  mainCanvas.style.width = w + "px";
  mainCanvas.style.height = h + "px";
  evolvingCanvas.width = w;
  evolvingCanvas.height = h;
  evolvingCanvas.style.width = w + "px";
  evolvingCanvas.style.height = h + "px";
  width = w;
  height = h;
}

// Set the canvas sizes to match the image dimensions, scaling based on the target size
function updateCanvasSizes(targetSize, imgW, imgH) {
  // Test the aspect ratio
  if (imgW > imgH) {
    setCanvasDim(targetSize, Math.round(targetSize * imgH / imgW));
  }
  else {
    setCanvasDim(Math.round(targetSize * imgW / imgH), targetSize)
  }
}


var canvasSize = Number(canvasSizeInput.value);
setCanvasDim(canvasSize, canvasSize);


var imgFileWidth, imgFileHeight, mainImg, mainImgData;

canvasSizeInput.addEventListener("change", (event) => {
  let num = Number(event.target.value);
  if (Number.isNaN(num)) {
    window.alert("canvas size must be a number");
    canvasSizeInput.value = canvasSize;
    return false;
  }
  if (!Number.isInteger(num)) {
    window.alert("canvas size must be an integer");
    canvasSizeInput.value = canvasSize;
    return false;
  }
  canvasSize = num;

  if (imgFileWidth && imgFileHeight) {
    updateCanvasSizes(canvasSize, imgFileWidth, imgFileHeight);
    ctx.drawImage(mainImg, 0, 0, width, height);
    mainImgData = ctx.getImageData(0, 0, width, height).data;
  }
});

learningRateInput.addEventListener("change", (event) => {
  let num = Number(event.target.value);
  if (Number.isNaN(num)) {
    window.alert("learning reat must be a number");
    learningRateInput.value = agent.learningRate;
    return false;
  }

  agent.learningRate = num;
});

imageFileInput.addEventListener("change", (event) => {
  var fr = new FileReader;

  fr.onload = function() {
    var img = new Image();
    img.onload = function() {
      mainImg = img;

      imgFileWidth = mainImg.width;
      imgFileHeight = mainImg.height;

      updateCanvasSizes(canvasSize, imgFileWidth, imgFileHeight);
      ctx.drawImage(mainImg, 0, 0, width, height);
      mainImgData = ctx.getImageData(0, 0, width, height).data;
    };

    img.src = fr.result;
  }

  fr.readAsDataURL(event.target.files[0]);
});

var interval;
startButton.addEventListener("click", () => {
  if (!mainImg) {
    window.alert("choose an image first");
    return false;
  }
  if (interval) {
    return false;
  }
  interval = window.setInterval(function() {
    agent.draw();
    agent.updateNetwork();
    document.querySelector("#generation").innerText = agent.generation;
  }, 0);
  agent.draw();
  startButton.setAttribute("disabled", "true");
  stopButton.removeAttribute("disabled");
});

stopButton.addEventListener("click", () => {
  window.clearInterval(interval);
  interval = null;
  startButton.removeAttribute("disabled");
  stopButton.setAttribute("disabled", "true");
  agent.reset();
});

function createAgentNN() {
  return new NN({
    layerSizes: [2, 300, 3],
    activationFunctions: [NN.LEAKY_RELU(0.1), NN.SIGMOID],
    wInit: {
      method: NN.RANDOM,
      range: 0.5,
    },
    bInit: {
      method: NN.RANDOM,
      range: 0.5,
    },
  });
}

var agent = {
  nn: createAgentNN(),
  inputScale: 1,
  dwTotal: null,
  dbTotal: null,
  learningRate: Number(learningRateInput.value),
  generation: 0,

  // Draws based on current network; also totals up parameter derivatives from backpropagation
  draw() {
    // Build derivative total arrays
    this.dwTotal = [];
    this.dbTotal = [];
    for (let l = 0; l < this.nn.numLayers - 1; l++) {
      let dw2d = [];
      for (let _ = 0; _ < this.nn.layerSizes[l + 1]; _++) {
        dw2d.push((new Array(this.nn.layerSizes[l])).fill(0));
      }
      this.dwTotal.push(dw2d);
      this.dbTotal.push((new Array(this.nn.layerSizes[l + 1])).fill(0));
    }
    
    let imgData = eCtx.createImageData(width, height);
    let data = imgData.data;
    let i = 0;
    for (let x = 0; x < width; x++) {
      let sx = (x / width - 0.5) * this.inputScale;
      for (let y = 0; y < height; y++) {
        let sy = (y / height - 0.5) * this.inputScale;
        let res = this.nn.feedForward([sx, sy]);

        // Backpropagate with the target values from the main image data
        let bp = this.nn.backpropagate([mainImgData[i] / 255, mainImgData[i + 1] / 255, mainImgData[i + 2] / 255]);

        // Add derivatives to the derivative totals
        this.dbTotal = NN.add2d(this.dbTotal, bp.b);
        this.dwTotal = NN.add3d(this.dwTotal, bp.w);

        data[i] = Math.round(res[0] * 255);
        data[i + 1] = Math.round(res[1] * 255);
        data[i + 2] = Math.round(res[2] * 255);
        data[i + 3] = 255;

        i += 4;
      }
    }
    eCtx.putImageData(imgData, 0, 0);
  },

  // Updates network based on derivative totals and learning rate
  updateNetwork() {
    // Add the learned derivatives, averaged over all pixels and multiplied by the LR
    this.nn.b = NN.add2d(this.nn.b, NN.mulScalar2d(this.dbTotal, -this.learningRate / (width * height)));
    this.nn.w = NN.add3d(this.nn.w, NN.mulScalar3d(this.dwTotal, -this.learningRate / (width * height)));
    this.generation++;
  },

  reset() {
    this.nn = createAgentNN();
    this.generation = 0;
  }
};
