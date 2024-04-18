# %pip install PyQt5

import yaml
import numpy as np
from sklearn.decomposition import PCA as skPCA
from vispy import app, scene, visuals


with open("rtvslo.yaml", "rt") as file:
    data = yaml.load(file, Loader=yaml.CLoader)

# print(data)

# l = 0
# for line in data:
#     l += len(line["gpt_keywords"])
# l /= len(data)
# print(l)
# # 20.22775...


# TF-IDF
print("Calculating TF-IDF...")

# tf
#   - raw count of a term in a document (kolikokrat se pojavi keyword pri nekem dokumentu - načeloma 1 tukaj)
#   - denominator : koliko je vse skupaj kljucnih besed pri dokumentu
#   - sepravi basically 1/(st klucnih besed)

# idf
#   - N : st dokumentov - len(data)
#   - number of documents where the term appears
#   - log N/(num of docs where appear)

# plan:
#   - cez vse clanke - vrstice v data
#   - v novo strukturo:
#       - id clanka (njegov index v data)
#       - dict kjer so kljuci kljucne besede in vrednosti slovar z tf, idf in tf-idf
#       - najprej notri tf
#   - v locen pomozni dict se daje kljucne besede in steje, kolikokrat se pojavijo
#   - se en obhod cez data, zdaj se v novi strukturi doda se idf in poracuna tf-idf

numAllDocs = len(data)
dataTfidf = [{} for i in range(numAllDocs)]
keywordsIdf = {}

#cez vse clanke
for i in range(len(data)):
    vrstica = data[i]
    # dataTfidf[i] = {}
    stKeywords = len(vrstica["gpt_keywords"])

    #cez kljucne besede
    for keyword in vrstica["gpt_keywords"]:

        if keyword in keywordsIdf.keys():
            keywordsIdf[keyword] += 1
        else:
            keywordsIdf[keyword] = 1

        if keyword in dataTfidf[i].keys():
            dataTfidf[i][keyword]["tf"] += 1
        else:
            dataTfidf[i][keyword] = {
                "tf" : 1,
                "idf" : None,
                "tf-idf" : None
            }
    
    for keyword in dataTfidf[i]:
        dataTfidf[i][keyword]["tf"] /= stKeywords

for i in range(len(data)):
    vrstica = data[i]
    for keyword in vrstica["gpt_keywords"]:
        dataTfidf[i][keyword]["idf"] = np.log(numAllDocs / keywordsIdf[keyword])
        dataTfidf[i][keyword]["tf-idf"] = dataTfidf[i][keyword]["tf"] * dataTfidf[i][keyword]["idf"]


# BoW... (bo sparse matrix)
print("Creating representations...")

chosenKeywords = [] # bag
for item in keywordsIdf.items():
    if item[1] >= 20:
        chosenKeywords.append(item[0])

bowRepresentations = [[] for i in range(numAllDocs)]
for i in range(len(dataTfidf)):
    vrstica = dataTfidf[i]
    vectorDict = {word : 0 for word in chosenKeywords}
    for keyword in vrstica.keys():
        vectorDict[keyword] = vrstica[keyword]["tf-idf"]
    vector = []
    for word in chosenKeywords:
        vector.append(vectorDict[word])
    bowRepresentations[i] = vector


# PCA
print("Reducing dimensionality...")

X = bowRepresentations

pca = skPCA(n_components=3)
pca.fit(X)
Y = pca.transform(X)

# Loadings
print("Processing loadings...")
loads = pca.components_.T 
loadsInfluences = np.linalg.norm(loads, axis=1)

# sns.heatmap(loads, annot=True, cmap='coolwarm', center=0)
# plt.show()

chosenLoadsIndxs = []
thresh = 0.8

for i in range(len(loads)):
    # original variable - keyword
    loading = loads[i]
    if loadsInfluences[i]/loadsInfluences.max() >= thresh:
        chosenLoadsIndxs.append(i)
    elif abs(loading[0]/np.max(loads, axis=0)[0]) > thresh:
        chosenLoadsIndxs.append(i)
    elif abs(loading[1]/np.max(loads, axis=0)[1]) > thresh:
        chosenLoadsIndxs.append(i)
    elif abs(loading[2]/np.max(loads, axis=0)[2]) > thresh:
        chosenLoadsIndxs.append(i)

chosenLoads = np.array([loads[i] for i in chosenLoadsIndxs])


# Visualization
print("Showing visualization...")

print(f"\nLoadings ({len(chosenLoads)}):")
for indx in chosenLoadsIndxs:
    print(f"{chosenKeywords[indx]} : {loads[indx]}")
print()

canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# projections
scatter = scene.visuals.Markers()
scatter.set_data(Y, edge_color=None, face_color=(0.36, 0.86, 1, 0.5), size=5)
view.add(scatter)

# origins = np.zeros((len(chosenLoads), 3))
# arrows = scene.visuals.Arrow(pos=origins, arrows=chosenLoads, arrow_size=10, arrow_color=(0.63, 0.13, 0.94, 1), arrow_type='triangle_60')
# view.add(arrows)

# loadings
origin = np.array([0, 0, 0])
line_data = []
for endpoint in chosenLoads:
    line_data.append(origin)
    line_data.append(endpoint)
line_data = np.array(line_data)
lines = scene.visuals.Line(pos=line_data, color=(0.63, 0.13, 0.94, 1), method='gl', connect='segments')
view.add(lines)

# loadings labels
for indx in chosenLoadsIndxs:
    label = scene.visuals.Text(text=chosenKeywords[indx], pos=loads[indx], color='white', font_size=10, parent=view.scene)

view.camera = 'turntable'

axis = scene.visuals.XYZAxis(parent=view.scene)

app.run()