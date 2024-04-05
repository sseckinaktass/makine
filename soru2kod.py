import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df = pd.read_excel("normalized_veriler.xlsx")  


X = df.drop(columns=['Class variable (0 or 1)'])  
y = df['Class variable (0 or 1)']  


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit(X, y).transform(X)


plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2
for color, i in zip(colors, [0, 1, 2]):
    plt.scatter(
        X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, lw=lw, label=f"Class {i}"
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of dataset")


plt.figure()
for color, i in zip(colors, [0, 1, 2]):
    plt.scatter(
        X_lda[y == i, 0], X_lda[y == i, 1], alpha=0.8, color=color, label=f"Class {i}"
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of dataset")

plt.show()
