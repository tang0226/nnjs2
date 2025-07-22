# 7/21/25

This experiment uses basic backpropagation to (ideally) imitate an image with a neural network.
The network takes the coordinates of a pixel as input and outputs the three channels for the resulting color.

## Log

After a quick grind to code this up yesterday, here are the roadblocks I ran into:
* Unoptimized CPU-side JS without WebWorkers (yet) would run slowly for any decent-sized image, so I used a size of 100px or less (so, small).
* Another problem was that, while the network could imitate general color patterns, it could not replicate details well

After playing with hyperparameters a bunch, I decided to tackle a simpler problem; instead of generating a full 2d image with 3 channels of color, a more general case could be to approximate a simple 1-variable function. This idea, which I coded earlier today in the `function-approximation` experiment, worked well and gave me the assurance that at least my backpropagation calculus is correct :)
