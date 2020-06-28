import torch
import numpy as np
from torch import nn, optim
from utility import process_image
from torchvision import models


from collections import OrderedDict

def new_model(architecture, hidden_units, droprate, learnrate):
    """
    Builds new network starting from pre-trained model, and two hidden + one output layer
    for the classifier.
    It defines the Loss function.
    It defines the optimizer using Adam with the defined learnrate.

    Parameters
    ----------
    architecture : string, the pre-trained model to start from, accepts only
                    densenet121, densenet169, densenet201 and vgg16
    hidden_units : integer, size of the first hidden layer
    droprate : float, drop-out rate used for all hidden layers
    learnrate : float, learning rate used for optimizer

    Returns
    -------
    model : the pre-trained model with modified classifier
    optimizer : the optimizer
    criterion : the loss function
    """
       
    if architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        layers = [1024, hidden_units, int(hidden_units / 2)]
    elif architecture == 'densenet169':
        model = models.densenet169(pretrained=True)
        layers = [1664, hidden_units, int(hidden_units / 2)]
    elif architecture == 'densenet201':
        model = models.densenet201(pretrained=True)
        layers = [1920, hidden_units, int(hidden_units / 2)]
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        layers = [25088, hidden_units, int(hidden_units / 10)]
    else:
        pass #TODO: we should fail here    
    
        
    # Freeze parameters of the pretrained model (we only want to train the classifier)
    for param in model.parameters():
        param.requires_grad = False
        
    classifier_structure = OrderedDict()
    
    for i in range(len(layers) - 1):
        classifier_structure.update({('fc' + str(i), nn.Linear(layers[i], layers[i + 1]))})
        classifier_structure.update({('relu' + str(i), nn.ReLU())})
        classifier_structure.update({('dropout' + str(i), nn.Dropout(droprate))})
    classifier_structure.update({('fc' + str(len(layers) - 1), nn.Linear(layers[-1], 102))})
    classifier_structure.update({('output', nn.LogSoftmax(dim=1))})


    model.classifier = nn.Sequential(classifier_structure)
    # Loss function
    criterion = nn.NLLLoss()
    
    # Optimizer, used to update weights with gradients
    optimizer = optim.Adam(model.classifier.parameters(), lr = learnrate)
    
    return model, optimizer, criterion
    
    
def train_model(model, optimizer, criterion, gpu, epochs, trainloader, validloader):
    """
    Train the new model on the training data set.
    After each epoch training progress is verified using the validation data
    set.

    Parameters
    ----------
    model : the model to train
    optimizer : optimizer function
    criterion : loss function
    gpu : True / False whether to use cuda or not
    epochs : number of epochs to train
    trainloader : dataloader for training data
    validloader : dataloader for validation data

    Returns
    -------
    None.

    """
    print("Start training network.")    
    if(gpu):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    
    model.to(device)
    
    for e in range(epochs):
        steps = 0
        running_loss = 0
        print(f"\nStarting epoch {e+1} of {epochs}.\n")
        for inputs, labels in trainloader:
            print('.', end='') # to display progress
            
            # Transfer inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset optimizer
            optimizer.zero_grad()
            # Forward pass
            logps = model.forward(inputs)
            # Calculate loss
            loss = criterion(logps, labels)
            running_loss += loss.item()
            steps += 1
            # Backpropagate loss
            loss.backward()
            # Update weights
            optimizer.step()
        
        # At end of epoch check running loss and accuracy on validation data
        test_loss = 0
        accuracy = 0
        # Turn off drop-out for validation
        model.eval()
 
        # Don't calculate gradients for validation (speed)
        with torch.no_grad():
            for inputs, labels in validloader:
                # Transfer inputs and labels to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                logps = model.forward(inputs)
                # Calculate loss
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            # Print statistics
            print(f"\n\nEpoch {e+1}:\n"
                  f"Train loss : {running_loss / steps:.2f}\n"
                  f"Validation loss : {test_loss / len(validloader):.2f}\n"
                  f"Validation accuracy : {accuracy / len(validloader):.2f}\n")
        # Turn dropout back on
        model.train()
    
def test_model(model, criterion, testloader, gpu):
    """
    Test accuracy of the model using test dataset.

    Parameters
    ----------
    model : the model to test
    criterion : the loss function
    testloader :  dataloader for validation data
    gpu : True / False whether to use cuda or not

    Returns
    -------
    None.

    """
    
    print("Start testing of the trained network.")    
    if(gpu):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'    
    model.to(device)
    accuracy = 0
    # Turn off drop-out for validation
    model.eval()
    test_loss = 0
    # Don't calculate gradients for testing (speed)
    with torch.no_grad():
        for inputs, labels in testloader:
            # Transfer inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            logps = model.forward(inputs)
            # Calculate loss
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        # Print statistics
        print(f"Test accuracy: {accuracy / len(testloader):.2f}\n")
    # Turn dropout back on
    model.train()
    
def save_checkpoint(model, architecture, optimizer, hidden_units, droprate, learnrate, epoch, train_data, file_path):
    """
    Save checkpoint for the model.

    Parameters
    ----------
    model : the model to save
    architecture : pre-trained model to use when restoring model
    optimizer: the optimizer function
    hidden_units : integer, size of the first hidden layer
    droprate : float, drop-out rate used for all hidden layers
    learnrate : float, learning rate used for optimizer
    epoch : epoch reached during training
    file_path : save location

    Returns
    -------
    None.

    """
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': architecture,
                  'hidden_units': hidden_units,
                  'droprate': droprate,
                  'learnrate': learnrate,
                  'epoch': epoch,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    torch.save(checkpoint, file_path)
    print("Checkpoint saved")
    
def load_checkpoint(checkpoint_path):
    """
    Load model from saved checkpoint.

    Parameters
    ----------
    checkpoint_path : location of checkpoint file

    Returns
    -------
    Returns
    -------
    model : the pre-trained model with modified classifier
    optimizer : the optimizer
    criterion : the loss function

    """
    checkpoint = torch.load(checkpoint_path)
    model, optimizer, criterion = new_model(checkpoint['architecture'],
                     checkpoint['hidden_units'],
                     checkpoint['droprate'],
                     checkpoint['learnrate'],
                    )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model, optimizer, criterion

def predict(model_path, image_path, top_k, gpu):
    """
    For image using checkpoint calculate most likely image classes and probability
    
    Parameters
    ----------
    model_path : model checkpoint to load
    image_path : path to image
    top_k : number of most likely image classes to return
    gpu : True / False whether to use cuda or not

    Returns
    -------
    probs : probabilities
    labels : labels
    """
    # Load model and set to evaluation mode
    model, optimizer, criterion = load_checkpoint(model_path)
    if(gpu):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu' 
    model.to(device)
    model.eval()

    # Prepare image
    image = process_image(image_path)
    # Tranfer image to tensor
    image = torch.from_numpy(np.array([image])).float()
    # Image to device
    image = image.to(device)
    
    # Feedforward through model
    logps = model.forward(image)
    ps = torch.exp(logps)
        
    # Define topk most likely classes and probabilities
    probs = ps.topk(top_k, dim=1)[0].tolist()[0]
    classes = ps.topk(top_k, dim=1)[1].tolist()[0]

    # Find labels
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
    labels = []
    for i in range(top_k):
        labels.append(ind[classes[i]])

    return probs, labels
