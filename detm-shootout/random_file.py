import pickle, gzip


with gzip.open("/home/zxia15/russian-semantics/detm-shootout/work/russian/xDETM_matrice.pkl.gz", 'rb') as f:
    while True:
        try:
            item = pickle.load(f)
            print(item.keys())
            del item
        except EOFError:
            break
