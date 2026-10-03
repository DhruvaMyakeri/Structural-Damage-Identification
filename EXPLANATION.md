# How this project works

The question is narrow. Given one photograph, is the structure damaged or not.

On the full test split of 8,000 photos, the best network we trained, MobileNetV2 with its last blocks unfrozen, gets **99.70%**. Chance, if you always guess the bigger class, is **50%**, because every split is exactly half undamaged and half damaged. The paper you were assigned reported **83%** for its best model. Both numbers are real. They are not in conflict once you look at which photographs each one was scored on.

## The goal, then the obvious fix, then the next problem

The goal, from Ho and Troncoso (CS229, 2018), is to label a photo 0 (undamaged) or 1 (damaged) so that after an earthquake you do not have to visit every building.

The obvious approach is to treat the photo as a list of numbers and hand that list to a classifier you already know: logistic regression, k-nearest neighbours, or a support vector machine. That is what the paper tried first, and what `src/classical.py` still does.

It breaks when the thing you care about can slide around the picture. A crack is a thin line. Flatten the image and that line becomes a handful of coordinates in a vector of length 3,072 (our 32 by 32 colour photos) or 150,528 (the paper's full 224 by 224 photos). Move the crack two centimetres and those coordinates change. Two photos of the "same" damage are far apart as vectors, so a neighbour search and a straight decision in pixel space have a hard time.

The fix is a model that looks at small patches and does not care where in the frame the patch sits. That is a convolutional network. The new problem is that we do not have millions of building photos, and a network the size of InceptionV3 will memorise a few thousand of them if you train every weight. The paper's repair, and ours, is transfer learning: start from a network already trained on ImageNet, freeze most of it, and train a small new layer for our two classes.

That repair creates its own problem. If the frozen network's features are a bad match for buildings, the new layer cannot get the accuracy up, no matter how long you train it. The paper saw that (their frozen Inception sat well below the training accuracy, which they called bias) and answered by unfreezing more layers. On our photos the frozen MobileNet is already at 99.45% test accuracy, so that second repair barely has anything left to fix. The sections below are that chain, with the numbers from `results/metrics.json` and `results/train.log`.

## What one photo becomes

A training photo in the zip is a JPEG, folder `Positive` (crack, label 1) or `Negative` (intact, label 0). `src/data.py` reads the zip in place.

Two different conversions happen after that.

```
Flat models (KNN, logistic regression, pixel SVM)
  JPEG
    -> resize to 32 by 32 RGB
    -> flatten to 3,072 numbers
    -> (pixel / 128) - 1     each value lands in [-1, 1)
    -> one row of the matrix X

Networks
  JPEG
    -> resize to 224 (MobileNetV2) or 299 (InceptionV3)
    -> divide by 255, then subtract ImageNet mean and divide by ImageNet std
    -> the frozen trunk
    -> a new linear layer with 2 outputs
    -> cross-entropy, which already includes the softmax
```

The paper used `(pixel / 128) - 1` for everything, because that is what their TensorFlow Inception preprocess did. Torchvision's ImageNet weights were trained with a different shift and scale (mean about 0.485, 0.456, 0.406 and std about 0.229, 0.224, 0.225). Feeding them the paper's formula would throw away the weights we downloaded. `src/cnn.py` uses the torchvision scaling on purpose. The flat models keep the paper's formula, in `pixels_to_vectors`.

## Why the neighbours look accurate and still miss half the cracks

K-nearest neighbours with k = 5, the paper's setting, scores **75.12%** on 4,000 test photos. That sounds like a mediocre but honest classifier. The confusion counts say something sharper.

|  | Predicted undamaged | Predicted damaged |
| --- | --- | --- |
| True undamaged | 2,000 | 0 |
| True damaged | 995 | 1,005 |

Every time it says "damaged", it is right. Precision on the damaged class is 100% (1,005 / 1,005). It finds only 1,005 of 2,000 cracks, so recall is 50.25%. Accuracy can be 75% while the model almost refuses to raise its hand.

That is what a neighbour vote does to a thin pattern. The five closest training vectors, in flattened pixel space, are often intact concrete, because a crack in a new position does not land near a crack in an old position. The vote goes to undamaged unless the crack happens to sit where a training crack sat. The paper's k = 5 model had the same shape of failure in a harsher form: 67.4% train, about 53% validation, close to always guessing the majority class. Ours is less collapsed because a 32 by 32 crack close-up is a simpler picture than a street photo of a building. The refusal-to-call-damaged pattern is still there. Accuracy alone would have hidden it. The counts are in `results/metrics.json` under the KNN entry, field `confusion_test`, in the order true-undamaged/predicted-undamaged, false damaged, missed crack, found crack.

## Where logistic regression stops climbing

Logistic regression with an L2 penalty and C = 1 (again the paper's setting) fits the 3,000 training rows well: **97.27%** train accuracy. On 4,000 held-out photos it is **87.90%**. The gap is about nine points. The paper's logistic regression was 99.4% on train and 54.7% on validation, which is memorising. Ours generalises, and it still leaves a large gap under the networks.

`learning_curve_logistic` in `src/classical.py` refits that same model on 300, 900, 1,500, 2,100, and all 3,000 training rows, and scores each fit on the same validation slice.

| Training rows | Train accuracy | Validation accuracy |
| --- | --- | --- |
| 300 | 100% | 76.33% |
| 900 | 99.78% | 80.75% |
| 1,500 | 99.13% | 82.03% |
| 2,100 | 98.33% | 81.45% |
| 3,000 | 97.27% | 81.75% |

Validation rises, then sits near 82%. Training accuracy falls as the set grows, which is what you expect: more points are harder to fit perfectly, and the fit that remains is a bit more honest. The curve says the limit is the representation, the flat 32 by 32 vector, not the sample size. Adding more crack photos in the same format will not climb from 82% to 99%. The figure is `results/figures/learning_curve_logreg.png`.

The pixel SVM (RBF kernel, C = 1, gamma = 0.001, the paper's constants) is the strongest flat model: **94.20%** train, **90.30%** validation, **92.97%** test, on the same 32 by 32 vectors. Gamma = 0.001 is not absurd at this resolution. A typical squared distance between two of these vectors is a few hundred, and 0.001 times a few hundred is order 1, so the kernel still has a slope. At the paper's full resolution the same gamma makes almost every pair look unrelated, which is one reason their SVM hit 99.7% train and 50.7% validation. We did not retune this gamma. The paper did not either.

## Transfer learning, in the code we actually ran

ImageNet is a thousand everyday categories. The paper points out that buildings are a tiny slice of it, so a network trained there is not a structural engineer. It is still a useful eye for edges, texture, and repetition. Transfer learning means: keep that eye, replace the last decision.

`build_mobilenet` in `src/cnn.py` loads MobileNetV2 with ImageNet weights, swaps the final linear layer for a layer with 2 outputs, and sets `requires_grad = False` on every convolutional weight. The log line from training is the check:

```
trainable 2,562 of 2,226,434
```

2,562 is 1,280 inputs times 2 outputs, plus 2 biases. That is the entire model we are fitting in the frozen run. Five epochs of Adam at learning rate 0.001, batch 64, on 6,000 training photos (3,000 per class). The weights kept are the epoch with the best validation accuracy, not the last epoch. `train_classifier` copies that state dict and reloads it before any test number is computed.

Frozen MobileNetV2, on all 8,000 test photos:

| Train | Validation | Test |
| --- | --- | --- |
| 99.63% | 99.29% | 99.45% |

Confusion: 3,981 true undamaged, 19 false damaged, 25 missed cracks, 3,975 found cracks.

InceptionV3 is the same idea with a heavier trunk. Frozen, it trains 4,098 numbers out of 25,116,362 (2,048 inputs times 2, plus biases). Four epochs, batch 16, images resized to 299. Test accuracy **98.46%**. The paper's best number was this architecture at 83% validation. Ours is higher, and it is not even our best network. Both facts come from the photos, which the next section separates from the method.

## Unfreezing did a small thing, because there was little bias left

The paper's frozen Inception was limited by bias: training and validation both sat in a band that more epochs of the same small layer would not leave. Their repair was to train more of the network, which adds parameters, which can then overfit.

We did that repair on MobileNetV2, because it is the network we can fine-tune quickly. After the frozen run, `unfreeze_last_mobilenet_blocks` turns gradients back on for the last four convolutional blocks, and training continues from those frozen-run weights for three epochs at learning rate 0.0001. Continuing matters. A brand new head trained only at 0.0001 would barely move, and that experiment would not match the paper.

```
trainable 1,528,642 of 2,226,434
```

Result on the same 8,000 test photos: **99.70%** test, **99.70%** validation, **99.97%** train. Confusion: 3,986, 14, 10, 3,990. Twenty-four mistakes in 8,000 photos, against forty-four for the frozen run.

So unfreezing helped. The help is a quarter of a percentage point. Train accuracy went to 99.97%, which is a hint of memorising, and validation did not fall, so it did not turn into the overfit the paper warned about. The paper's story (frozen model stuck, extra layers needed, then overfitting to be fought with random images) does not play out here. The frozen model was not stuck.

## Why flips and shifts made the test number worse

A third MobileNet, trained from the ImageNet trunk again with the same frozen setup, sees a horizontal flip and a shift of up to 12% of the frame on every batch. The paper also tried zoom, and their poster shows zoomed frames that are no longer believable structures, so we left zoom out.

Test accuracy **99.03%**. Validation **99.26%**. Train, measured on the original photos with augmentation turned off, **99.17%**.

That is worse than the frozen run's 99.45% test. Augmentation is supposed to help when the model is memorising. This model was not memorising. The flipped and shifted copies are still almost the same crack, so they do not teach a new distinction, and they make the small trained layer's job slightly noisier. The paper reported that a careful augmentation raised validation by about one point while a harsher one mainly fought overfit. We measured a small drop. That is the result. It is not a reason to claim augmentation is generally harmful.

## The SVM on a middle layer, and a gamma grid that never turned down

The paper's last model does not use the thousand-way ImageNet decision. It takes an earlier activation, where the features are more generic, and fits an SVM. Their TensorFlow graph had a layer numbered 288 out of 311. Torchvision does not number layers that way. `src/hybrid.py` hooks the block named `Mixed_6e`, which still has `Mixed_7a`, `Mixed_7b`, and `Mixed_7c` after it, and replaces the spatial map with its average. Each photo becomes 768 numbers. The log confirms the shape: `train features (2000, 768)`.

Gamma (how narrow the RBF kernel is) is chosen on validation, the way their Figure 3 does. We tried six values.

| Gamma | Validation accuracy |
| --- | --- |
| 0.0001 | 98.50% |
| 0.0003 | 98.50% |
| 0.001 | 99.12% |
| 0.003 | 99.71% |
| 0.01 | 99.83% |
| 0.03 | 99.88% |

0.03 wins, and it is the largest value in the grid. The curve is still rising. **Established:** 0.03 is the best of these six. **Not established:** that a larger gamma would be worse. We did not run one. The figure is `results/figures/gamma_sweep.png`.

On its own test slice of 2,400 photos this SVM scores **99.96%**: 1,199 true undamaged, 1 false damaged, 0 missed cracks, 1,200 found cracks. One mistake. `train.py` selects the demo model by validation accuracy alone, so this SVM is what `python demo.py` loads. Its validation accuracy is 99.88% on 2,400 photos. Unfrozen MobileNet's is 99.70% on 8,000. Those are different denominators. A 0.18 point edge on a smaller slice does not prove the SVM is the better model. What the selection rule did is written down in `results/best.json`. What the full 8,000-photo test supports as the best network is the unfrozen MobileNet, at 99.70%.

The number `demo.py` prints next to a label for this model is the SVM margin for the predicted class, not a probability. On four test photos (two of each class, seed 7) the margins were about 1.4 to 1.5 and all four labels matched the truth. A margin is "how far past the boundary." It is not "the chance this is right."

## What the paper did, and what this repository changed

| | Paper | This run |
| --- | --- | --- |
| Photos | PEER Hub ImageNet, 5,913, wide shots and close-ups, clutter, some bad labels | Ozgenel concrete cracks, 40,000, close-ups only, balanced |
| Why not PEER | They had the challenge files | The public copy is emailed after a form. Not a direct download |
| Split | 90/10, and they called the 10% both validation and test | The zip's own train / validation / test. Test is not used to pick the demo model |
| Flat input | Full frame, flattened | 32 by 32, then the same `(pixel / 128) - 1` |
| Flat training size | About 5,300 | 1,500 per class (3,000 rows) |
| KNN k=5 | Train 67.4%, validation about 53.7% | Train 79.73%, validation 68.90%, test 75.12% |
| Logistic, C=1 | Train 99.4%, validation 54.7% | Train 97.27%, validation 81.60%, test 87.90% |
| RBF SVM, gamma 0.001 | Train 99.7%, validation 50.7% | Train 94.20%, validation 90.30%, test 92.97% |
| Network | MobileNetV1 and InceptionV3 in TensorFlow | MobileNetV2 and InceptionV3 in PyTorch |
| Frozen Inception | Validation 83% | Test 98.46%, validation 98.78% |
| MobileNet | Validation about 67% | Frozen test 99.45%. Unfrozen test 99.70% |
| Mid-layer SVM | Their layer 288, validation 75% | Mixed_6e, 768-d, test 99.96% on 2,400 photos |
| Optimiser | Gradient descent at 0.01, later Adam because the curves were noisy | Adam, 0.001 frozen, 0.0001 once blocks are unfrozen |

The method list matches the paper. The dataset does not, and that is the gap that explains the table. Crack close-ups do not contain people, curtains, telephone lines, or a collapsed floor. A crack against grey concrete is a much cleaner signal than the paper's photos, so every model moves up, and the flat models move up enough that "classical methods are stuck at chance" is false here.

```
Paper photos                         Our photos
wide shot or a patch of wall         a tile of concrete
crack, spall, or a fallen building   a crack, or no crack
people, branches, curtains           almost nothing else
some labels wrong                    labels follow the folder
classical models near 50%            classical models 75% to 93%
Inception clearly ahead              MobileNet slightly ahead
```

## The files, in the order a run uses them

`download_data.py` fetches three zips from the Hugging Face mirror if they are not already in `data/zips/`. Training does not unpack them. Forty thousand tiny files are slow to write on Windows, and `open_image` in `src/data.py` reads a member straight out of the zip.

`train.py` is the entry point.

1. `load_splits` counts 12,000 / 12,000 train, 4,000 / 4,000 validation, 4,000 / 4,000 test.
2. `stratified_take` keeps a seeded subset. Seed 42 for the flat models, 43 for the networks, 44 for the feature SVM, so the three experiments do not silently share one draw.
3. `pixels_to_vectors` builds the 32 by 32 matrices. Flat models then fit in `src/classical.py`.
4. `run_cnn` builds a loader, trains, reloads the best validation weights, and only then computes train, validation, and test accuracy. Test is not inside the loop that picks the checkpoint.
5. The unfrozen MobileNet is the same module object continued, not a second network from scratch.
6. `inception_mixed6e_features` runs the pretrained Inception, not the fine-tuned one. The SVM is meant to see ImageNet features, which is the paper's setup.
7. Figures and `results/metrics.json` are written at the end. `results/best.json` records which checkpoint the demo loads, and why (highest validation accuracy).

`demo.py` branches on `arch` in that json. `hybrid_svm` extracts Mixed_6e and calls the saved SVM. A plain network path would load a state dict and print a softmax. A flat model would resize to 32 by 32. Using the flat path on the feature SVM would feed 3,072 numbers into a model that expects 768, so that branch is separate on purpose.

`src/metrics.py` is only the accuracy and the four confusion counts. `src/plots.py` draws the bars, the logistic curve, the epoch curves, the gamma sweep, and the winning confusion matrix.

## What these numbers license, and what they do not

**Established.** The pipeline above is what ran. The percentages in the table are the ones in `results/metrics.json`, from the log that ends with `wrote ... metrics.json`.

**Measured here.** Unfrozen MobileNetV2 test accuracy 99.70% on 8,000 photos. Frozen MobileNetV2 99.45% on the same 8,000. Pixel SVM 92.97% on 4,000. Feature SVM 99.96% on 2,400. KNN confusion as in the table above. Logistic validation flat near 82% after 1,500 rows.

**Supported by that measurement.** On balanced concrete close-ups, a model that keeps spatial structure beats a flat pixel vector. The step from the pixel SVM (about 93% on 4,000 photos) to the unfrozen MobileNet (99.70% on 8,000) is that claim. Chance is 50%, and every model clears it. Unfreezing the last MobileNet blocks helps by a small amount and does not collapse validation. Flip-and-shift augmentation does not help in this run.

**Not supported.** That we reproduced the paper's 83%, or that InceptionV3 is the right network for structural damage. Inception is behind MobileNet here by about one point, on an easier dataset. That classical models are useless for this task. They are weaker than the networks, and they are far from chance. That the feature SVM is definitely better than MobileNet. It was selected on a smaller validation slice, and its test slice is 2,400 photos rather than 8,000. That gamma 0.03 is optimal. It is the best value we tried. That these accuracies would hold on the paper's photos of collapsed buildings, clutter, and mislabelled frames. We never ran those photos. That more epochs, or training the whole of Inception, would matter. The frozen MobileNet was already at the ceiling of this dataset.

The experiment that would earn the paper's claim on our code is the same training script pointed at PEER Task 2 (damage state), with the same held-out test split. Until that file arrives by email, the honest sentence is the narrower one.
