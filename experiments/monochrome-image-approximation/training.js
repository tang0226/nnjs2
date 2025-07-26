importScripts("../../nn.js");

var nn, targetData;

this.onmessage = function(event) {
  let data = event.data;

  switch (data.type) {
    case "nn":
      nn = new NN({ nn: data.nn });
      break;
    
    case "targetData":
      targetData = data.targetData;
      break;

    case "weights":
      nn.w = data.w;
      break;

    case "biases":
      nn.b = data.b;
      break;

    case "train":
      let { trials, batchSize, width, height } = data;
      nn.startIteration();
      for (let i = 0; i < trials; i++) {
        // sample a random point on the canvas
        let x = Math.floor(Math.random() * width);
        let y = Math.floor(Math.random() * height);
        nn.feedForward([x / width, y / height]);
        nn.backpropagate([targetData[y * width + x]]);
      }
      this.postMessage({
        type: "done",
        // Sum of gradients over all trials; NOT AVERAGE
        gradientTotals: {
          w: nn.dwTotal,
          b: nn.dbTotal,
        },
        trials: trials,
        batchSize: batchSize,
      });
  }
};
