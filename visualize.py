import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
  file_name = sys.argv[1]
  data = pd.read_csv(file_name)
  index = data.columns.tolist()

  epoch_list = list(data[index[0]])
  acc_list = list(data[index[1]])
  val_loss_list = list(data[index[2]])
  train_loss_list = list(data[index[3]])

  fig, ax1 = plt.subplots()
  plt.title("Learning log")
  ax1.set_xlabel('epoch')
  ax1.set_ylabel('Loss')
  ax1.plot(val_loss_list, color="blue", label="Val-Loss")
  ax1.plot(train_loss_list, color="green", label="Train-Loss")

  ax2 = ax1.twinx()
  ax2.plot(acc_list, color="red", label="Accuracy")
  ax2.set_ylabel('Accuracy [%]')

  h1, l1 = ax1.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax2.legend(h1+h2, l1+l2)

  plt.show()