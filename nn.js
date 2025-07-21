class NN {
  constructor(obj) {
    if (obj.nn) {
      let nn = obj.nn;
      this.layerSizes = [...nn.layerSizes];
      this.numLayers = nn.numLayers;
      this.inputLayerSize = nn.inputLayerSize;
      this.outputLayerSize = nn.outputLayerSize;

      this.activationFunctions = this.af = [...nn.af];
      this.weights = this.w = nn.w.map((arr2d) => NN.copy2d(arr2d));
      this.weightDerivatives = this.dw = nn.dw.map((arr2d) => NN.copy2d(arr2d));
      this.biases = this.b = copy2d(nn.b);
      this.biasDerivatives = this.db = copy2d(nn.db);
      this.neuronOutputs = this.z = copy2d(nn.z);
      this.neuronOutputDerivatives = this.dz = copy2d(nn.dz);
      this.activations = this.a = copy2d(nn.a);
      this.activationDerivatives = this.da = copy2d(nn.da);
      return;
    }

    // layers
    this.layerSizes = obj.layerSizes;
    if (obj.layerSizes < 2) {
      throw new Error("ValueError: Network cannot have fewer than 2 layers");
    }

    this.numLayers = obj.layerSizes.length;
    this.inputLayerSize = obj.layerSizes[0];
    this.outputLayerSize = obj.layerSizes[this.numLayers - 1];

    // Activation functions
    if (obj.activationFunctions || obj.af) {
      let af = obj.activationFunctions || obj.af;
      if (af.length > obj.layerSizes - 1) {
        throw new Error("ValueError: Wrong number of activation functions");
      }
      else if (af.length == obj.numLayers - 1) {
        this.af = af;
      }
      else {
        // [hidden layers, output layer]
        if (af.length == 2) {
          this.af = new Array(this.numLayers - 2);
          this.af.fill(af[0]);
          this.af[this.numLayers - 2] = af[1];
        }
        // [all layers]
        else if (af.length == 1) {
          this.af = new Array(this.numLayers - 1);
          this.af.fill(af[0]);
        }
        else {
          throw new Error("ValueError: Wrong number of activation functions");
        }
      }
    }

    this.activationFunctions = this.af;

    // Weights [layerFrom][nTo][nFrom]
    this.weights = this.w = [];
    this.weightDerivatives = this.dw = [];

    if (obj.wInit) {
      if (obj.wInit.method) {
        if (obj.wInit.method == NN.RANDOM) {
          if (obj.wInit.range) {
            this.wInitFunc = function() {
              return Math.random() * obj.wInit.range * 2 - obj.wInit.range;
            }
          }
          else {
            throw new Error("Error: Random weight init method requires a range parameter");
          }
        }
        else {
          throw new Error("Error: unrecognized weight init method");
        }
      }
    }
    else {
      this.wInitFunc = function() {return 0};
    }

    for (let i = 0; i < this.numLayers - 1; i++) {
      this.w.push(NN.init2d(this.layerSizes[i + 1], this.layerSizes[i], this.wInitFunc));
      this.dw.push(NN.init2d(this.layerSizes[i + 1], this.layerSizes[i], () => 0));
    }

    // Biases [nonInputLayer][n]
    this.biases = this.b = [];
    this.biasDerivatives = this.db = [];

    if (obj.bInit) {
      if (obj.bInit.method) {
        if (obj.bInit.method == NN.RANDOM) {
          if (obj.bInit.range) {
            this.bInitFunc = function() {
              return Math.random() * obj.bInit.range * 2 - obj.bInit.range;
            }
          }
          else {
            throw new Error("Error: Random bias init method requires a range parameter");
          }
        }
        else {
          throw new Error("Error: unrecognized bias init method");
        }
      }
    }
    else {
      this.bInitFunc = function() {return 0};
    }

    for (let i = 0; i < this.numLayers - 1; i++) {
      let l = [];
      for (let j = 0; j < this.layerSizes[i + 1]; j++) {
        l.push(this.bInitFunc());
      }
      this.b.push(l);

      let dl = new Array(this.layerSizes[i + 1]);
      dl.fill(0);
      this.db.push(dl);
    }

    // outputs (z = sum(prevA * weight) + bias; a = activationFunc(z))
    this.neuronOutputs = this.z = [];
    this.neuronOutputDerivatives = this.dz = [];
    for (let i = 0; i < this.numLayers - 1; i++) {
      this.z.push((new Array(this.layerSizes[i])).fill(0));
      this.dz.push((new Array(this.layerSizes[i])).fill(0));
    }

    // Activations [nonInputLayer][n]
    this.activations = this.a = [];
    this.activationDerivatives = this.da = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.a.push((new Array(this.layerSizes[i])).fill(0));

      // da will not apply to the input layer; a is the only array with numLayers elements;
      // All others have numLayers - 1
      if (i != 0) {
        this.da.push((new Array(this.layerSizes[i])).fill(0));
      }
    }
  }


  feedForward(inputs) {
    for (let i = 0; i < this.layerSizes[0]; i++) {
      this.a[0][i] = inputs[i];
    }

    for (let lFrom = 0; lFrom < this.numLayers - 1; lFrom++) {
      for (let iTo = 0; iTo < this.layerSizes[lFrom + 1]; iTo++) {
        let ws = 0;
        for (let iFrom = 0; iFrom < this.layerSizes[lFrom]; iFrom++) {
          ws += this.a[lFrom][iFrom] * this.w[lFrom][iTo][iFrom];
        }
        ws += this.b[lFrom][iTo];
        this.z[lFrom][iTo] = ws;
        this.a[lFrom + 1][iTo] = this.af[lFrom].func(ws);
      }
    }

    return [...this.a[this.numLayers - 1]];
  }

  backpropagate(y, inputs = []) {
    if (inputs.length) {
      this.feedForward(inputs);
    }

    // LAST LAYER

    // Last layer da (uses the loss function (y - a) ^ 2)
    for (let i = 0; i < this.outputLayerSize; i++) {
      this.da[this.numLayers - 2][i] = 2 * (this.a[this.numLayers - 1][i] - y[i]);
    }

    // Last layer dz, db, and dw
    let li = this.numLayers - 2;
    for (let iTo = 0; iTo < this.outputLayerSize; iTo++) {
      // dz value for this neuron
      let _dz = this.da[li][iTo] * this.af[li].dFunc(this.z[li][iTo]);
      this.dz[li][iTo] = _dz;
      
      // b is a linear constant in z, so db = dz
      this.db[li][iTo] = _dz

      // dw values for this neuron
      for (let iFrom = 0; iFrom < this.layerSizes[li - 1]; iFrom++) {
        this.dw[li][iTo][iFrom] = _dz * this.a[li][iFrom];
      }
    }

    li--;

    // HIDDEN LAYERS

    for (; li > -1; li--) {
      for (let iTo = 0; iTo < this.layerSizes[li + 1]; iTo++) {
        // da
        // Loop through neurons that the current neuron signals to
        let _da = 0;
        for (let i = 0; i < this.layerSizes[li + 2]; i++) {
          _da += this.dz[li + 1][i] * this.w[li + 1][i][iTo];
        }
        this.da[li][iTo] = _da;

        // dz and db
        let _dz = _da * this.af[li].dFunc(this.z[li][iTo]);
        this.dz[li][iTo] = _dz;
        this.db[li][iTo] = _dz;

        // dw:
        // Here, technically, this.layerSizes[li] and this.a[li] are using
        // the index (li - 1 + 1), since we are going back a layer, but
        // these arrays include the input layer, so indices must be shifted forward
        for (let iFrom = 0; iFrom < this.layerSizes[li]; iFrom++) {
          this.dw[li][iTo][iFrom] = _dz * this.a[li][iFrom];
        }
      }
    }

    return {
      w: this.dw,
      b: this.db,
      z: this.dz,
      a: this.da
    };
  }



  static RANDOM = "RANDOM";
  static XAVIER = "XAVIER";
  static HE = "HE";

  static SIGMOID = {
    func(x) {
      return 1 / (1 + Math.exp(-x))
    },
    dFunc(x) {
      let s = this.func(x);
      return s * (1 - s);
    },
    name: "sigmoid"
  };

  static RELU = {
    func(x) {
      return Math.max(0, x);
    },
    dFunc(x) {
      return x > 0 ? 1 : 0;
    },
    name: "relu"
  };

  static LEAKY_RELU(alpha = 0.01) {
    return {
      alpha: alpha,
      func(x) {
        return Math.max(x, this.alpha * x);
      },
      dFunc(x) {
        return x > 0 ? 1 : this.alpha;
      },
      name: "leaky relu"
    };
  }

  static zero2d(a, b) {
    let res = [];
    for (let i = 0; i < a; i++) {
      let arr = new Array(b);
      arr.fill(0);
      res.push(arr);
    }
    return res;
  }

  static init2d(a, b, func) {
    let res = [];
    for (let i = 0; i < a; i++) {
      let arr = [];
      for (let j = 0; j < b; j++) {
        arr.push(func());
      }
      res.push(arr);
    }
    return res;
  }

  static copy2d(arr2d) {
    return arr2d.map((arr) => [...arr]);
  }

  static add2d(a, b) {
    return a.map((arr, i) => arr.map(
      (n, j) => n + b[i][j]
    ));
  }

  static add3d(a, b) {
    return a.map((arr, i) => NN.add2d(arr, b[i]));
  }

  static mulScalar2d(arr, n) {
    return arr.map((a) => a.map((x) => x * n));
  }

  static mulScalar3d(arr, n) {
    return arr.map((a) => a.map((b) => b.map((x) => x * n)));
  }
}
