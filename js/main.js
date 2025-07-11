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

var game = new _2048();

var tileMargin = 10;
var tileSize = 100;
var fontSize = tileSize * 0.5;
var colors = {
  boardBorder: "#b9ae9f",
  tiles: [
    "#cbc0b3", // empty
    "#ece4da", // 2
    "#ebe1ca", // 4
    "#eab477", // 8
    "#ea9762", // 16
    "#e87b5f", // 32
    "#e85e3d", // 64
    "#e6d36a", // 128
    "#e7d058", // 256
    "#e6cd42", // 512
    "#e5c92b", // 1024
    "#e4c505", // 2048
    "#b182b6", // 4096
    "#a259ae", // 8192
    "#9a3ca7", // 16384
    "#790084", // 32768
    "#5a0048", // 65536
    "#8f7ee7", // 131072
  ],
};


var _canvasSideLength = tileSize * game.gridSize + tileMargin * (game.gridSize + 1);
setCanvasDim(_canvasSideLength, _canvasSideLength);




function up() {game.move(_2048.UP)}
function down() {game.move(_2048.DOWN)}
function right() {game.move(_2048.RIGHT)}
function left() {game.move(_2048.LEFT)}
