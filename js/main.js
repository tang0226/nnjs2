const mainCanvas = document.getElementById("main-canvas");
const ctx = mainCanvas.getContext("2d");

var width, height;
function setCanvasDim(w, h) {
  mainCanvas.width = w;
  mainCanvas.height = h;
  mainCanvas.style.width = w + "px";
  mainCanvas.style.height = h + "px";
  width = w;
  height = h;
}

setCanvasDim(1000, 1000);

var nn = new NN({
  layerSizes: [4, 6, 3, 4],
  af: [NN.RELU, NN.SIGMOID],
  wInit: {
    method: NN.RANDOM,
    range: 2,
  },
  bInit: {
    method: NN.RANDOM,
    range: 2,
  },
});

console.log(nn);
console.log(nn.feedForward([2, 3, 4, 5]));