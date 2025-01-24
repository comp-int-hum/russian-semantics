import argparse, logging, numpy, json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

logger = logging.getLogger('generate_auth2auth_img')


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument("-ia", '--id2auth_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-f', "--filtered_author", type=str, required=True)

    args = parser.parse_args()

    author_of_interest = args.filtered_author.split(';')

    a2a_mat = numpy.loadtxt(args.input_dir, delimiter=',')

    with open(args.id2auth_dir, 'r') as f:
        id2auth = json.load(f)
    
    id2auth = {int(k) : v for k, v in id2auth.items()}
    auth2id = {v : k for k, v in id2auth.items()}
    filter_id = sorted([auth2id[author] for author in author_of_interest])
    filter_auth = [id2auth[idx].split(', ')[0] for idx in filter_id]
    filter_a2a_mat = numpy.round(a2a_mat[numpy.ix_(filter_id, filter_id)], 4)
    logger.info(f"reduced from original matrix size of {a2a_mat.shape} to {filter_a2a_mat.shape}")

    fig, ax = plt.subplots(figsize=(20, 10))
    table = ax.table(cellText=filter_a2a_mat, 
                     colLabels=filter_auth, 
                     rowLabels=filter_auth,
                     loc='center', 
                     cellLoc='center')
    
    for idx in range(len(filter_auth)):
        cell = table.get_celld().get((0, idx))  # +1 for header offset
        if cell is not None:
            cell.set_text_props(
                fontproperties=FontProperties(weight='bold'),
            )
        cell = table.get_celld().get((idx + 1, -1))  # +1 for header offset
        if cell is not None:
            cell.set_text_props(
                fontproperties=FontProperties(weight='bold'),
            )
    
    for row_idx in range(filter_a2a_mat.shape[0]):
        # Create a masked array to ignore the diagonal element
        row_data = filter_a2a_mat[row_idx].copy()
        row_data[row_idx] = numpy.inf  # Mask diagonal value
        
        # Find the minimum value and its position
        min_val = numpy.min(row_data)
        min_col = numpy.argmin(row_data)
        
        # Get the cell and highlight it
        cell = table.get_celld().get((row_idx + 1, min_col))  # +1 for header offset
        if cell is not None:
            cell.set_text_props(
                fontproperties=FontProperties(weight='bold'),
                color='red'
            )
            # cell._text.set_text(f'{min_val:.4f}') 

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # Add padding to cells
    for cell in table._cells.values():
        cell.PAD = 0.05
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output_dir, bbox_inches='tight', dpi=300)
    plt.close()