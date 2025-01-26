import gzip, json, pickle

with gzip.open("/home/zxia15/russian-semantics/detm-shootout/work/russian/xDETM_matrice.pkl.gz", 'rb') as f:
    data = pickle.load(f)

print(data.keys())