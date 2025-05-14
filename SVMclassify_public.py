import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

# Load the data
file_path = '../data/data_pub.csv'
data = pd.read_csv(file_path)

# user controls
features = ['BM1','BM3']
yscale = 'log'
xscale = 'linear'
X = data[features].values
y = data['Dx'].values


# Train SVM svm_classifier
svm_classifier = make_pipeline(
    StandardScaler(),
    SVC(
        C=1,
        kernel="poly",
        degree=2,
        coef0=1,
        gamma=1,
        probability=True,
        class_weight={0:1, 1:1},
        tol=1e-7,
        decision_function_shape='ovr'
    )
)

svm_classifier.fit(X, y)
y_prob = svm_classifier.predict_proba(X)[:, 1]
y_pred = svm_classifier.predict(X)

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), 
                     np.linspace(y_min, y_max, 1000))

Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#SVM Plotting
cmap_background = ListedColormap(['#85B56E', '#C6878C']) 
cmap_points = ListedColormap(['green', 'red'])
plt.figure(figsize=(5, 3), dpi=300)
plt.gca().set_facecolor('#708D9C')
plt.gca().patch.set_facecolor('#708D9C') 
plt.contourf(xx, yy, Z, alpha=1, cmap=cmap_background)
plt.gcf().set_facecolor('#708D9C')
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap_points, edgecolor='black', linewidth=0.5)

# Custom legend for Dx values
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='green', label='Dx = 0'),
    mpatches.Patch(color='red', label='Dx = 1')
]
plt.legend(handles=legend_handles, fontsize="xx-small", markerscale=0.8, edgecolor='black', loc='lower right')

plt.title('SVM Decision Boundary')
plt.xlabel(f'{features[0]}, ({'log' if xscale=='log' else ''} a.u.)')
plt.ylabel(f'{features[1]}, ({'log' if yscale=='log' else ''} a.u.)')
plt.yscale(yscale)
plt.xscale(xscale)
plt.xlim(-0.01, 1.05)
plt.ylim(0, 1.)
plt.subplots_adjust(bottom=0.2) 
plt.show()

#ROC Curve Plotting
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = roc_auc_score(y, y_prob)

plt.figure(figsize=(3, 3))
plt.gca().set_facecolor('#708D9C')
plt.gca().patch.set_facecolor('#708D9C')
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='#4CFFF1', linewidth=4)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.gca().set_facecolor('#708D9C')
plt.gca().patch.set_facecolor('#708D9C')
plt.gcf().set_facecolor('#708D9C')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#misclassification percentage
y_pred = svm_classifier.predict(X)
misclassification_percentage = (1 - accuracy_score(y, y_pred)) * 100
print(f'Misclassification Percentage: {misclassification_percentage:.2f}%')