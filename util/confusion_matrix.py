import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
NUM_CLASS = 7
acc = np.loadtxt("./checkpoint_file_4of5/result.csv",delimiter=',').astype(np.int)
pred, true = acc[:,0], acc[:,1]
array = [[0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]]
# confusion matrix
# col is truth
# row is pred
total_truth = 0
total_corr = 0
for i in range(NUM_CLASS):
    for j in range(NUM_CLASS):
        idx_t = true == i+1
        idx_p = pred == j+1
        idx = np.all([idx_t,idx_p],axis=0)
        # print(np.sum(idx),"out of",np.sum(idx_t))
        num_idx = np.sum(idx)
        num_idx_t = np.sum(idx_t)
        if i == j:
            total_truth = total_truth + num_idx_t
            total_corr = total_corr + num_idx
        array[j][i] = num_idx/num_idx_t*100
df_cm = pd.DataFrame(array, index = [i for i in "1234567"],
                  columns = [i for i in "1234567"])
plt.figure(figsize = (10,7))
cmap = sn.cm.rocket_r
sn.heatmap(df_cm, annot=True,cmap=cmap,annot_kws={"size": 12})
print("ACC: ",total_corr/total_truth*100)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t)
plt.show()