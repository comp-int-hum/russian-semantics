import argparse, numpy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)

    parser.add_argument('-m', '--mode', type=str, default='js')

    args = parser.parse_args()
 
    assert args.input_dir.endswith('.csv') and args.output_dir.endswith('.csv')

    in_data = numpy.loadtxt(args.input_dir, delimiter=',')
    num_auth = in_data.shape[0]

    if args.mode == 'js':
        out_data = numpy.zeros((num_auth, num_auth))

        for auth1 in range(num_auth):
            for auth2 in range(auth1, num_auth):
                data1, data2 = in_data[auth1], in_data[auth2]
                avg = 0.5 * (data1 + data2)
                js = 0.5 * (jensenshannon(data1, avg) ** 2 + jensenshannon(data2, avg) ** 2)
                out_data[auth1][auth2] = js
                out_data[auth2][auth1] = js
    
    elif args.mode == 'cos':
        out_data = cosine_similarity(in_data)
    
    elif args.mode == 'euc':
        out_data = euclidean_distances(in_data)
    
    numpy.savetxt(args.output_dir, out_data, delimiter=',')