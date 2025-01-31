import logging, json, argparse, numpy, pickle
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from lib.gapstat import GapStatClustering

logger = logging.getLogger('generate_auth2auth_cluster')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True) 
    parser.add_argument("-ia", "--id2auth", type=str, required=True)
    parser.add_argument("-r", "--data_root", type=str, required=True)
    parser.add_argument("-do", "--output_dir", type=str, default="generate_auth2auth_cluster.txt")
    parser.add_argument("-f", "--filter_auth", type=str, required=True)
    parser.add_argument("-w", "--which_to_fit", type=str, default="all")
    parser.add_argument('-n', '--num_cluster', type=int, default=None)
    
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
    model = (KMeans(n_clusters=args.num_cluster,
                    n_init="auto").fit(auth_embed if args.which_to_fit == "all" else filter_auth_embed)
            if args.num_cluster 
            else GapStatClustering().fit(auth_embed if args.which_to_fit == "all" else filter_auth_embed)
            )

    logger.info("completes training KMeans model .... ")
            
    labels = model.labels_
    filter_labels = labels[filter_id] if args.which_to_fit == 'all' else labels
    unique_labels = numpy.unique(filter_labels)
    num_cluster = args.num_cluster if args.num_cluster else model.n_clusters_
    logger.info(f"producing a KMeans model with {num_cluster} clusters")
    all_output_text = ''
    for cluster_idx in range(num_cluster):
        cluster_points = (auth_embed[labels == cluster_idx] if args.which_to_fit == 'all' 
                        else filter_auth_embed[labels == cluster_idx])
        cluster_centroid = numpy.mean(cluster_points, axis=0)
        cluster_variance = numpy.mean(numpy.var(cluster_points, axis=0))
        # cluster_authors = [auth for idx, auth in zip(filter_id, filter_auth) if labels[idx] == cluster_idx]
        cluster_authors = ([filter_auth[idx] for idx, filter_idx in enumerate(filter_id) if labels[filter_idx] == cluster_idx]
                           if args.which_to_fit == 'all'
                           else [auth for idx, auth in enumerate(filter_auth) if labels[idx] == cluster_idx])
        cluster_info = f"Cluster {cluster_idx}: \n Centroid {cluster_centroid}\n Variance {cluster_variance}\n " +  '\n'.join(cluster_authors) + "\n"
        all_output_text += cluster_info + "\n"
    
    logger.info(f"writing cluster data to {args.output_dir}")
    with open(args.output_dir, 'w') as f:
        f.write(all_output_text)

    models = []
    logger.info("starts fitting dimension collapse .... ")
    models.append((f"{args.data_root}/images/generate_auth2auth_cluster_tSNE_{args.which_to_fit}.png", 
                TSNE(n_components=2, learning_rate='auto',
                init='random', perplexity=3).fit_transform(
                    auth_embed if args.which_to_fit == 'all' else filter_auth_embed)))

    models.append((f"{args.data_root}/images/generate_auth2auth_cluster_PCA_{args.which_to_fit}.png", 
        decomposition.PCA(n_components=2).fit_transform(
        auth_embed if args.which_to_fit == 'all' else filter_auth_embed)))

    # transform_auth_embed = transform_auth_embed[filter_id]
    logger.info("completes fillting dimension collapse .... ")
    for output_dir, transform_auth_embed in models:
        plt.figure()
        if args.which_to_fit == 'all':
            transform_auth_embed = transform_auth_embed[filter_id]
        for label in unique_labels:
            auth_per_label = transform_auth_embed[filter_labels == label]
            plt.scatter(auth_per_label[:, 0], auth_per_label[:, 1], label=f'Group {label}')

        plt.legend()
        plt.xlabel('Reduced Topic Embed Dim 1')
        plt.ylabel('Reduced Topic Embed Dim 2')
        plt.title('Author Clustering by Topic Distribution')
        logger.info(f"storing image to {output_dir}")
        plt.savefig(output_dir)