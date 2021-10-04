# MAD613-TEXTURE SIMILARITY METRICS COMPARISON
## What is done?
#### We carried out a novel evaluation using Maximum differentiation (MAD) method between texture similarity metrics on visual textures. In total, three models are chosen to be compared, containing metrics from image quality assessment domain and model from texture synthesis domain. We use MAD method to make comparison between all the chosen models using a constrained optimization process. A subjective test is conducted to evaluate the results from the MAD competition, the aggressive matrix, resistance matrix and global scores for all metric are computed based on the subjective test.

## What is Maximum differentiation (MAD)
#### MAD is a method used to efficiently compare different models (Gram, SSIM, MSE in this case). Firstly, we create a initial distorted image. Next, we keep the output of one model same (Defender) while maximizing/minimizing another model (Attacker). We got two pairs of images at one initial point, since models will take turn being attacker and defender. Finally, we carry out a subjective test to see which model is better.

## Results:
![alt text](https://i.imgur.com/i4NzSLX.png)
#### Example: SSIM successfully falsify Gram model since the pair of image SSIM created is visually distinguishable. Also, in this case SSIM successfully defence the attack from Gram model since the pair of image Gram model created is not visually distinguishable
