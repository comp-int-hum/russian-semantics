import logging, json, argparse, numpy, pickle
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logger = logging.getLogger('generate_auth2auth_cluster')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True) 
    parser.add_argument("-ia", "--id2auth", type=str, required=True)
    parser.add_argument("-io", "--img_output_dir", type=str, default="images/generate_auth2auth_cluster.png")
    parser.add_argument("-do", "--data_output_dir", type=str, default="generate_auth2auth_cluster.txt")
    parser.add_argument("-f", "--filter_auth", type=str, required=True)
    parser.add_argument('-n', '--num_cluster', type=int, default=5)
    parser.add_argument('-c', '--dim_collapse', type=str, default='PCA')
    
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    auth_of_interest = args.filter_auth.split(';')
    auth_embed = numpy.loadtxt(args.input_dir, delimiter=',')
    
    with open(args.id2auth, 'r') as f:
        id2auth = json.load(f)
    
    id2auth = {int(k) : v for k, v in id2auth.items()}
    auth2id = {v : k for k, v in id2auth.items()}
    filter_id = sorted([auth2id[author] for author in auth_of_interest])
    filter_auth = [id2auth[idx] for idx in filter_id]
    filter_auth_embed = auth_embed[filter_id]

    logger.info(f"reducing original embedding of size {auth_embed.shape} to {filter_auth_embed.shape}")
    logger.info("starts training KMeans model .... ")
    kmeans = KMeans(n_clusters=args.num_cluster,
                    random_state=0,
                    n_init="auto").fit(filter_auth_embed)

    logger.info("completes training KMeans model .... ")
            
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    all_output_text = ''
    for cluster_idx in range(kmeans.n_clusters):
        cluster_points = filter_auth_embed[labels == cluster_idx]
        cluster_variance = numpy.mean(numpy.var(cluster_points, axis=0))
        # cluster_authors = [auth for idx, auth in zip(filter_id, filter_auth) if labels[idx] == cluster_idx]
        cluster_authors = [auth for idx, auth in enumerate(filter_auth) if labels[idx] == cluster_idx]
        cluster_info = f"Cluster {cluster_idx}: \n Centroid {centroids[cluster_idx]}\n Variance {cluster_variance}\n " +  '\n'.join(cluster_authors) + "\n"
        all_output_text += cluster_info + "\n"
    
    logger.info(f"writing cluster data to {args.data_output_dir}")
    with open(args.data_output_dir, 'w') as f:
        f.write(all_output_text)

    logger.info("starts fitting dimension collapse .... ")
    if args.dim_collapse == 'tSNE':
        transform_auth_embed = TSNE(n_components=2, learning_rate='auto',
                     init='random', perplexity=3).fit_transform(filter_auth_embed)
    elif args.dim_collapse == 'PCA':
        transform_auth_embed = decomposition.PCA(n_components=2).fit_transform(filter_auth_embed)
    else:
        raise Exception(f"dimension collapse method {args.dim_collpase} unimplemented")

    # transform_auth_embed = transform_auth_embed[filter_id]
    logger.info("completes fillting dimension collapse .... ")
    # filter_labels = labels[filter_id]
    filter_labels = labels
    unique_labels = numpy.unique(labels)
    for label in unique_labels:
        auth_per_label = transform_auth_embed[filter_labels == label]
        plt.scatter(auth_per_label[:, 0], auth_per_label[:, 1], label=f'Group {label}')

    plt.legend()
    plt.xlabel('Reduced Topic Embed Dim 1')
    plt.ylabel('Reduced Topic Embed Dim 2')
    plt.title('Author Clustering by Topic Distribution')
    logger.info(f"storing image to {args.img_output_dir}")
    plt.savefig(args.img_output_dir)