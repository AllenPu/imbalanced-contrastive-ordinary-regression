import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


labels_ce = np.load('./acc/labels_ce.npy')
labels_la = np.load('./acc/labels_la.npy')
labels_soft_label = np.load('./acc/labels_soft_label.npy')
pred_ce = np.load('./acc/pred_ce.npy')
pred_la = np.load('./acc/pred_la.npy')
pred_soft_label = np.load('./acc/pred_soft_label.npy')


length = labels_ce.shape[0]
diff_ce = {}
diff_la = {}
diff_soft = {}
for i in range(length):
    soft_ab = int(np.abs(pred_soft_label[i]-labels_soft_label[i]))
    diff_soft[soft_ab] = diff_soft.get(soft_ab, 0) + 1
    ce_ab = int(np.abs(pred_ce[i]-labels_ce[i]))
    diff_ce[ce_ab] = diff_ce.get(ce_ab, 0) + 1
    la_ab = int(np.abs(pred_la[i]-labels_la[i]))
    diff_la[la_ab] = diff_la.get(la_ab, 0) + 1
list_soft = [diff_soft[key] for key in sorted(diff_soft)]
list_ce = [diff_ce[key] for key in sorted(diff_ce)]
list_la = [diff_la[key] for key in sorted(diff_la)]



colors = ['r', 'g', 'b']
yticks = ['soft','ce', 'la']
df = pd.DataFrame({'soft': np.array(list_soft), 'ce': np.array(list_ce), 'la':np.array(list_la)}, columns=['soft', 'ce','la'])
df.plot.hist(alpha=0.8,bins=length)
plt.savefig('diff.jpg')
#plt.show()