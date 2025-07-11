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


class _2048Agent {
  constructor(obj) {
    this.nn = obj.nn;
    this.game = obj.game;
  }

  act() {
    if (this.game.gameOver) {
      return false;
    }
    let a = this.nn.feedForward(this.game.grid.reduce((a = [], b) => a.concat(b)));
    let choices = [
      {choice: _2048.UP, val: a[0]},
      {choice: _2048.RIGHT, val: a[1]},
      {choice: _2048.DOWN, val: a[2]},
      {choice: _2048.LEFT, val: a[3]},
    ];

    choices.sort((a, b) => a.val < b.val ? 1 : (b.val < a.val ? -1 : 0));

    let i = 0;
    while (i < 4) {
      if (this.game.move(choices[i].choice) != -1) {
        break;
      }
      i++;
    }
    this.score = this.game.score;
    if (game.gameOver) {
      this.done = true;
    }
    return true;
  }
}


var tileMargin = 10;
var tileSize = 100;
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
  twoFourFont: "#726551",
  otherTileFont: "#ffffff",
};

var fontSizeCoeffs = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.35, 0.35, 0.35, 0.35, 0.3, 0.3, 0.3, 0.275];


function drawGame(game) {
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = colors.boardBorder;
  ctx.fillRect(0, 0, width, height);

  for (let y = 0; y < game.gridSize; y++) {
    let tileCornerY = tileMargin + (tileSize + tileMargin) * y;
    for (let x = 0; x < game.gridSize; x++) {
      let tileCornerX = tileMargin + (tileSize + tileMargin) * x;
      let tile = game.grid[y][x];
      ctx.fillStyle = colors.tiles[tile];
      ctx.fillRect(tileCornerX, tileCornerY, tileSize, tileSize);

      if (tile > 0) {
        ctx.fillStyle = (tile == 1 || tile == 2) ? colors.twoFourFont : colors.otherTileFont;
        ctx.font = `${tileSize * fontSizeCoeffs[tile]}px Arial`;
        let text = _2048.POW2[tile].toString();
        let textMetrics = ctx.measureText(text);
        let textWidth = textMetrics.width;
        let textHeight = textMetrics.actualBoundingBoxAscent - textMetrics.actualBoundingBoxDescent;
        ctx.fillText(
          text,
          tileCornerX + tileSize / 2 - textWidth / 2,
          tileCornerY + tileSize / 2 + textHeight / 2
        );
      }
    }
  }

  document.querySelector("#score").innerText = game.score;
  document.querySelector("#turns").innerText = game.turns;
}


var nn = new NN({
  layerSizes: [16, 32, 32, 4],
  af: [NN.LEAKY_RELU(), NN.SIGMOID],
  wInit: {
    method: NN.RANDOM,
    range: 2,
  },
  bInit: {
    method: NN.RANDOM,
    range: 2,
  },
});

var game = new _2048({
  gridSize: 4
});

var agent = new _2048Agent({
  nn: nn,
  game: game,
});


var _canvasSideLength = tileSize * game.gridSize + tileMargin * (game.gridSize + 1);
setCanvasDim(_canvasSideLength, _canvasSideLength);


function draw() {
  for (let i = 0; i < 100; i++) {
    if (agent.act()) {
      drawGame(game);
    }
    else {
      document.querySelector("#alert").innerText = "Game Over!";
      return false;
    }
  }

  setTimeout(draw, 0);
}

document.querySelector("#restart").addEventListener("click", () => {
  if (game.gameOver) {
    nn = new NN({
      layerSizes: [16, 32, 32, 4],
      af: [NN.LEAKY_RELU(), NN.SIGMOID],
      wInit: {
        method: NN.RANDOM,
        range: 2,
      },
      bInit: {
        method: NN.RANDOM,
        range: 2,
      },
    });
    game.reset();
    agent = new _2048Agent({
      nn: nn,
      game: game,
    });
    document.querySelector("#alert").innerText = "";
    draw();
  }
});

draw();

/*
document.addEventListener("keydown", (event) => {
  if (!game.gameOver) {
    switch (event.key) {
      case "ArrowUp":
        if (game.up() != -1) {
          draw();
        }
        break;
      case "ArrowDown":
        if (game.down() != -1) {
          draw();
        }
        break;
      case "ArrowRight":
        if (game.right() != -1) {
          draw();
        }
        break;
      case "ArrowLeft":
        if (game.left() != -1) {
          draw();
        }
        break;
      case " ":
        agent.act();
        draw();
        break;
    }
    if (game.gameOver) {
      document.querySelector("#alert").innerText = "Game Over!";
    }
  }

});
*/
