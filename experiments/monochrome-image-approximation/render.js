importScripts("../../nn.js");

var nn, targetData;

this.onmessage = function(event) {
  let data = event.data;

  switch (data.type) {
    case "nn":
      // Create a full NN instance from the serialized network passed
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
    
    case "render":
      let {yStart, chunkHeight, chunkI, width, height} = data;
      let yEnd = yStart + chunkHeight;
      let imgDataArr = new Uint8ClampedArray(4 * width * chunkHeight);

      let i = 0;
      for (let y = yStart; y < yEnd; y++) {
        let ny = y / height;
        for (let x = 0; x < width; x++) {
          let nx = x / width;
          let bw = Math.round(nn.feedForward([nx, ny])[0] * 255);
          imgDataArr[i++] = bw;
          imgDataArr[i++] = bw;
          imgDataArr[i++] = bw;
          imgDataArr[i++] = 255;
        }
      }

      this.postMessage({
        type: "done",
        imgDataArr: imgDataArr,
        y: yStart,
        chunkI: chunkI,
      });

      break;
  }
}
