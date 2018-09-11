import argparse
from data_utils import load_label_mapping
from model_utils import Network
import torch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_image', action='store',
                        help='Pass the image (/w path)')
    
    parser.add_argument('checkpoint_file', action='store', 
                        help='Pass the checkpoint (/w path)')

    parser.add_argument('--top_k', action='store',
                        dest='top_k', default = 1,
                        help='Returns the top K most likely classes', type = int)

    parser.add_argument('--category_names', action='store',
                        dest='category_names', default ='cat_to_name.json',
                        help='Choose the pre-trained architecture: vgg11 or vgg13')

    parser.add_argument('--gpu', action='store_true',
                        default=False, dest='use_gpu',
                        help='Use GPU for training')

    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.0')

    results = parser.parse_args()

    net = Network()

    device = torch.device("cuda:0" if results.use_gpu else "cpu")

    net.set_device(device)

    list_probs, list_predsclass = net.predict(results.path_to_image, results.checkpoint_file, topk = results.top_k)

    cat_to_name = load_label_mapping(results.category_names)
    
    if results.top_k == 1:
        print(list_probs, cat_to_name[list_predsclass])
    else:
        print(list_probs, [cat_to_name[item] for item in list_predsclass])

if __name__ == "__main__":
    main()