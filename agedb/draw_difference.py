import numpy as np
import matplotlib.pyplot as plt


labels_ce = np.load('labels_ce.npy')
labels_la = np.load('labels_la.npy')
labels_soft_label = np.load('labels_soft_label.npy')
pred_ce = np.load('pred_ce.npy')
pred_la = np.load('pred_la.npy')
pred_soft_label = np.load('pred_soft_label.npy')


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


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ['r', 'g', 'b']
yticks = ['soft','ce', 'la']
x = [i for i in range(20)]
for c, k in zip(colors, yticks):
    xs = x
    ys = np.array(f'list_{k}')
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)


ax.set_xlabel('groups')
ax.set_ylabel('criteria')
ax.set_zlabel('Z')


ax.set_yticks(yticks)
plt.show()