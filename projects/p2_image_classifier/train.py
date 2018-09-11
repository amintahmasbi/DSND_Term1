import argparse
from data_utils import load_data
from model_utils import Network
import torch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', action='store',
                        help='Set the data directory')
    
    parser.add_argument('--save_dir', action='store',
                        dest='save_dir', default = '.',
                        help='Set directory to save checkpoints')

    parser.add_argument('--lr','--learning_rate', action='store',
                        dest='learning_rate', default = None,
                        help='Set the learning rate for the optimizer', type = float)

    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units', default = 512,
                        help='Set the number of nodes in the hidden layer', type = int)

    parser.add_argument('-e','--epochs', action='store',
                        dest='epochs', default = 5,
                        help='Set the learning epochs', type = int)

    parser.add_argument('--arch', action='store',
                        dest='arch', default = 'vgg11',
                        help='Choose the pre-trained architecture: vgg11 or vgg13')

    parser.add_argument('--gpu', action='store_true',
                        default=False, dest='use_gpu',
                        help='Use GPU for training')

    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.0')

    results = parser.parse_args()

    trainloader, validloader, testloader, class_to_idx = load_data(results.data_dir)

    net = Network()
    
    device = torch.device("cuda:0" if results.use_gpu else "cpu")
    net.set_device(device)

    model = net.build_model(hidden_units = results.hidden_units, arch = results.arch)
    
    print(model)
    
    model.class_to_idx = class_to_idx



    net.compile_model(results.learning_rate)

    chkpnt_file_name = 'checkpoint_part2.pth'

    checkpoint_file = results.save_dir + '/' + chkpnt_file_name

    net.train(trainloader, validloader, epochs=results.epochs)

    net.save_checkpoint(checkpoint_file, results.epochs)

if __name__ == "__main__":
    main()