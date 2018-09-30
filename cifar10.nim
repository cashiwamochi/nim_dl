import os
import streams, sequtils, random
import arraymancer


const 
  CIFAR10Filenames = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
    "test_batch.bin"
    ]

  height = 32
  width = 32
  c = 3
  num_patches_in_file = 10000
  num_data_in_row = 3073

type 
  cifar10 = tuple[
    train_images: Tensor[uint8],
    test_images: Tensor[uint8],
    train_labels: Tensor[uint8],
    test_labels: Tensor[uint8]
    ]

  cifar10Temp = tuple[
    images: Tensor[uint8],
    labels: Tensor[uint8]
    ]

  patchImageData = array[0 .. c-1, array[0 .. height-1, array[0 .. width-1, uint8]]]

proc checkCifarExist(cifar_dir: string, cifar_file_names: array[0..5, string]) : bool = 
  for file_name in cifar_file_names:
    if not existsFile(cifar_dir & "/" & file_name):
      return false
  return true

proc read_cifar10*(cifar10_dir: string, file_names: seq[string]): cifar10Temp = 
  var
    cifar10_image_patches = newSeq[patchImageData](10000*len(file_names)) # patch * c * h * w
    cifar10_labels = newSeq[uint8](10000*len(file_names)) # batch * 1

  for file_idx, file_name in file_names:
    var fs = newFileStream(cifar10_dir & "/" & file_name, fmRead)
    if not isNil(fs):
      var 
        col_idx = 0 # 0 - 3072
        row_idx = 0 # 0 - 9999
      let
        # offset = num_data_in_row*file_idx
        offset = num_patches_in_file*file_idx
      while not fs.atEnd:
        let value : uint8 = fs.readUint8()
        if col_idx == 0: # 0
          cifar10_labels[row_idx + offset] = value
        elif col_idx <= 1024: # 1 - 1024 r
          cifar10_image_patches[row_idx + offset][0][(col_idx-1) div width][(col_idx-1) mod width] = value
        elif col_idx <= 2048: # 1025 - 2048 g
          cifar10_image_patches[row_idx + offset][1][(col_idx-1025) div width][(col_idx-1025) mod width] = value
        elif col_idx <= 3072: # 2049 - 3072 b
          cifar10_image_patches[row_idx + offset][2][(col_idx-2049) div width][(col_idx-2049) mod width] = value
        else:
          echo "|| [ERROR] Ploblem in parsing"
          break
  
        col_idx = col_idx + 1
  
        if col_idx == num_data_in_row:
          col_idx = 0
          row_idx = row_idx + 1
      
      assert row_idx == num_patches_in_file
    fs.close()
  
  # This shuffle makes the performance better
  randomize(1)
  shuffle(cifar10_image_patches)
  randomize(1)
  shuffle(cifar10_labels)

  result.images = cifar10_image_patches.toTensor()
  result.labels = cifar10_labels.toTensor()

proc load_cifar10*(cifar10_dir: string): cifar10 =
  if not checkCifarExist(cifar10_dir, CIFAR10Filenames):
    echo "Not found cifar10 ..."
    quit(1)

  # train
  let train_cifar: cifar10Temp = read_cifar10(cifar10_dir, CIFAR10Filenames[0..4])

  # test
  let test_cifar: cifar10Temp = read_cifar10(cifar10_dir, @[CIFAR10Filenames[5]])

  result.train_images = train_cifar.images
  result.train_labels = train_cifar.labels

  result.test_images = test_cifar.images
  result.test_labels = test_cifar.labels