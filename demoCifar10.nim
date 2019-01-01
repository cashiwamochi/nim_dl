import os
import streams, sequtils
import math, random, times
import arraymancer
import parsecsv
import cifar10

# main
echo "||================================= Cifar-10 (Nim) ================================= "
let
  root_path : string = os.getAppDir()
  now = getLocalTime(getTime())
  data = loadCifar10(root_path & "/cifar10")
  x_train = data.train_images.astype(float32) / 255.0'f32
  y_train = data.train_labels.astype(int)
  x_test = data.test_images.astype(float32) / 255.0'f32
  y_test = data.test_labels.astype(int)

createDir(root_path & "/log")
var log_content = "epoch, accuracy, val-loss, train-loss \n"

randomize(111)

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
    cv1_1:      Conv2D(x.out_shape, 32, 3, 1)
    cv1_2:      Conv2D(cv1_1.outshape, 32, 3, 2)
    mp1:        MaxPool2D(cv1_2.outshape, (2,2), (0,0), (2,2))
    cv2_1:      Conv2D(mp1.out_shape, 64, 3, 1)
    cv2_2:      Conv2D(cv2_1.out_shape, 64, 3, 2)
    mp2:        MaxPool2D(cv2_2.outshape, (2,2), (0,0), (2,2))
    fl:         Flatten(mp2.out_shape)
    hidden:     Linear(fl.out_shape, 500)
    classifier: Linear(500, 10)
  forward x:
    x.cv1_1.relu.cv1_2.relu.mp1.cv2_1.relu.cv2_2.relu.mp2.fl.hidden.relu.classifier

let 
  model = ctx.init(DemoNet)
var 
  optim = model.optimizerSGD(learning_rate = 0.01'f32)

# Learning loop
echo "|| Learning Start !"
for epoch in 0 .. 20:
  # Back-Propagation-Part
  if epoch == 5:
      optim = model.optimizerSGD(learning_rate = 0.005'f32)
  elif epoch == 10:
      optim = model.optimizerSGD(learning_rate = 0.002'f32)
  elif epoch == 15:
      optim = model.optimizerSGD(learning_rate = 0.001'f32)

  var sum_train_loss = 0.0
  for batch_id in 0 ..< data.train_images.shape[0] div n:
    let 
      offset = batch_id * n
      batch_data_x = X_train[offset ..< offset+n]
      batch_data_y = y_train[offset ..< offset+n]
      clf = model.forward(batch_data_x)
      train_loss = clf.sparse_softmax_cross_entropy(batch_data_y)

    sum_train_loss = sum_train_loss + train_loss.value.data[0]

    discard """
    if batch_id mod 100 == 0:
      echo "|| Epoch is: " & $epoch
      echo "|| Batch id: " & $batch_id
      echo "|| Loss is:  " & $loss.value.data[0]
    """

    train_loss.backprop()
    optim.update()

  # Validation
  ctx.no_grad_mode:

    var 
      score = 0.0
      val_loss = 0.0
    for i in 0 ..< 100:
      let y_pred = model.forward(X_test[i*100 ..< (i+1)*100]).value.softmax.argmax(axis = 1).squeeze
      score += accuracy_score(y_test[i*100 ..< (i+1)*100], y_pred)
      val_loss += model.forward(X_test[i*100 ..< (i+1)*100]).sparse_softmax_cross_entropy(y_test[i*100 ..< (i+1)*100]).value.data[0]
    score /= 100.0
    val_loss /= 100.0

    echo "||############## Epoch " & $epoch & " done. ##############"
    echo "|| Accuracy:     " & $(score * 100.0) & "%"
    echo "|| Val-Loss:     " & $val_loss
    echo "|| Train-Loss:   " & $(sum_train_loss / float(data.train_images.shape[0] div n))

    log_content = log_content & $epoch & "," & $(score * 100.0) & "," & $val_loss & "," & $(sum_train_loss / float(data.train_images.shape[0] div n)) & "\n"

  if epoch mod 2 == 0:
    writeFile(root_path & "/log/" & $now.hour & $now.minute & $now.second & "_" & $epoch & ".csv", log_content)

writeFile(root_path & "/log/" & $now.hour & $now.minute & $now.second & "_last.csv", log_content)
