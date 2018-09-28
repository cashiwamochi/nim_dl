import os
import streams, sequtils
import math, random
import arraymancer

import cifar10

# main
echo "||================================= Cifar-10 (Nim) ================================= "
let
  root_path : string = os.getAppDir()
  data = loadCifar10(root_path & "/cifar10")
  x_train = data.train_images.astype(float32) / 255.0'f32
  y_train = data.train_labels.astype(int)
  x_test = data.test_images.astype(float32) / 255.0'f32
  y_test = data.test_labels.astype(int)

echo "|| Arraymancer environment is being arranged ... "

randomize(8) # Random seed for reproducibility

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  n = 32                           # Batch size
  c = 3
  h = 32
  w = 32

  X_train = ctx.variable x_train
  X_test = ctx.variable x_test

# Configuration of the neural network
network ctx, DemoNet:
  layers:
    x:          Input([c, h, w])
    cv1:        Conv2D(x.out_shape, 20, 3, 3)
    mp1:        MaxPool2D(cv1.out_shape, (2,2), (0,0), (2,2))
    cv2:        Conv2D(mp1.out_shape, 50, 3, 3)
    mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
    cv3:        Conv2D(mp2.out_shape, 80, 3, 3)
    mp3:        MaxPool2D(cv3.out_shape, (2,2), (0,0), (2,2))
    fl:         Flatten(mp3.out_shape)
    hidden:     Linear(fl.out_shape, 500)
    classifier: Linear(500, 10)
  forward x:
    x.cv1.relu.mp1.cv2.relu.mp2.cv3.relu.mp3.fl.hidden.relu.classifier

let 
  model = ctx.init(DemoNet)
  optim = model.optimizerSGD(learning_rate = 0.001'f32)

# Learning loop
echo "|| Learning Start !"
for epoch in 0 ..< 30:
  # Back-Propagation-Part
  for batch_id in 0 ..< data.train_images.shape[0] div n:
    let 
      offset = batch_id * n
      batch_data_x = X_train[offset ..< offset+n]
      batch_data_y = y_train[offset ..< offset+n]
      clf = model.forward(batch_data_x)
      loss = clf.sparse_softmax_cross_entropy(batch_data_y)

    if batch_id mod 100 == 0:
      echo "|| Epoch is: " & $epoch
      echo "|| Batch id: " & $batch_id
      echo "|| Loss is:  " & $loss.value.data[0]

    loss.backprop()
    optim.update()

  # Validation
  ctx.no_grad_mode:
    echo "|| Epoch #" & $epoch & " done. Testing accuracy"

    var score = 0.0
    var loss = 0.0
    for i in 0 ..< 100:
      let y_pred = model.forward(X_test[i*100 ..< (i+1)*100]).value.softmax.argmax(axis = 1).squeeze
      score += accuracy_score(y_test[i*100 ..< (i+1)*100], y_pred)
      loss += model.forward(X_test[i*100 ..< (i+1)*100]).sparse_softmax_cross_entropy(y_test[i*100 ..< (i+1)*100]).value.data[0]
    score /= 100.0
    loss /= 100.0
    echo "|| Accuracy: " & $(score * 100.0) & "%"
    echo "|| Loss:     " & $loss
