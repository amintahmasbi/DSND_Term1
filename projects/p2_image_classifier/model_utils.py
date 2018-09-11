from torchvision import models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
import numpy as np
from PIL import Image
from data_utils import process_image

class Network():

    def __init__(self):
        self.input_size = 25088
        self.dropout_prob = 0.2
        self.output_size = 102
        self.hidden_size = 512

    def set_device(self, device = 'cpu'):
        self.device = device

    def build_model(self, hidden_units = 512, arch = 'vgg11'):

        if arch == 'vgg13':
            self.model = models.vgg13(pretrained=True)
        elif arch == 'vgg11':
            self.model = models.vgg11(pretrained=True)
        else: # TODO: Error case handling
            self.model = models.vgg11(pretrained=True) # or any default model


        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = hidden_units

        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(self.input_size, self.hidden_size)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout1', nn.Dropout(p=self.dropout_prob)),
                                  ('fc2', nn.Linear(self.hidden_size, self.output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
            
        self.model.classifier = classifier

        return self.model

    # A function to set criterion, optimizer, and other hyper-prameters
    def compile_model(self, learning_rate = None):
        
        if learning_rate:
            self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.classifier.parameters())

        self.criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        

    # A function for the validation pass
    def validation(self, testloader):
        test_loss = 0
        accuracy = 0

        self.model.to(self.device)

        # Model in inference mode, dropout is off
        self.model.eval()   

        for images, labels in testloader:

            images, labels = images.to(self.device), labels.to(self.device)

            output = self.model.forward(images)
            test_loss += self.criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        # Make sure dropout and grads are on for training
        self.model.train()
        
        return test_loss, accuracy

        # A function for the training the model
    def train(self, trainloader, validloader, epochs = 5, 
              checkpoint_file = None, print_every = 40, save_every = 5):
        
        steps = 0
        running_loss = 0    

        self.model.to(self.device)

        for e in range(epochs):
            # Model in training mode, dropout is on
            self.model.train()
            for images, labels in trainloader:
                
                # Move input and label tensors to the GPU
                images, labels = images.to(self.device), labels.to(self.device)
                steps += 1
                
                self.optimizer.zero_grad()
                
                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()

                if steps % print_every == 0:
                    
                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        valid_loss, accuracy = self.validation(validloader)
                    
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                          "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
                    
                    running_loss = 0
                    
            if checkpoint_file and ((e+1) % save_every == 0):
                # Save the checkpoint
                self.save_checkpoint(checkpoint_file, e)


    def save_checkpoint(self, checkpoint_file, epochs = 5):
        
        # Save the checkpoint
        checkpoint = {'input_size': self.input_size,
          'output_size': self.output_size,
          'hidden_size': self.hidden_size,
          'class_to_idx': self.model.class_to_idx,
          'ephocs_trained': epochs,
          'optimizer_state_dict': self.optimizer.state_dict(),
          'model_state_dict': self.model.state_dict()}

        torch.save(checkpoint, checkpoint_file)

    # A function that loads a checkpoint and rebuilds the model
    def load_checkpoint(self, checkpoint_file, learning_rate = None):
        checkpoint = torch.load(checkpoint_file)
        
        self.model = models.vgg11(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
            
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        output_size = checkpoint['output_size']

        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(input_size, hidden_size)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout1', nn.Dropout(p=0.2)),
                                  ('fc2', nn.Linear(hidden_size, output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        
        self.model.classifier = classifier

        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.class_to_idx = checkpoint['class_to_idx']

        self.compile_model(learning_rate)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return self.model

    def predict(self, image_path, checkpoint_file, topk = 5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Implement the code to predict the class from an image file
        self.model = self.load_checkpoint(checkpoint_file)
        
        pil_image = Image.open(image_path)
        np_image = process_image(pil_image)
        
        tensor_image = torch.from_numpy(np.expand_dims(np_image,axis=0)).type(torch.FloatTensor)
        tensor_image = tensor_image.to(self.device)
        #   model = load_checkpoint(model)
        self.model.idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()

            output = self.model.forward(tensor_image)

            ps = torch.exp(output)
            probs, preds = ps.topk(topk, dim=1)
            list_probs = probs.cpu().numpy().squeeze().tolist()
            list_preds = preds.cpu().numpy().squeeze().tolist()
            if topk == 1:
                list_predsclass = self.model.idx_to_class[list_preds]
            else:
                list_predsclass = [ self.model.idx_to_class[index] for index in list_preds ]
            
        return list_probs, list_predsclass