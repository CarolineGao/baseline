# Image Captioninng
# Pretrained CNN - remove the last output softmax layer of pretrained CNN. - LSTM, sequence to sequence
# Send the output from CNN to 2 ways: start token or hidden dim

from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models

# CNN 
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN  # We are not going to train CNN model, we use the pretrained CNN model, fine-tune the last layer. 
        self.inception = models.inception_v3(pretrained=True, aux_logits=False) # CNN model we use is inception model which is called GoogleNetv3, a famous ConvNet trained on Imagenet from 2015. 
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size) # fc.in_features is number of input for your linear layer. embed_size is the output feature. in_features–size of each input sample; out_features–size of each output sample.
        # The above layer, in_features is the output from cnn which is inception layer. The output feature is embed_size which will be the input of RNN. 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        features = self.inception(images)  # cnn the images
        # we do not train the entire CNN, we only calculate the gradient for the last layer. 
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:  # if fully connected weight and fully connected bias in name. 
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN # self.train_CNN = False at the moment, if we want to train the entire CNN, we can set it as True. 
        return self.dropout(self.relu(features))


# RNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # M x N matrix, with M being the number of words and N being the size of each word vector. 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)     # initialize LSTM
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0) 
        # torch.cat ((x, y), dim=0)基于row堆叠，把features + captions. 
        # unsqueeze(0) is to add additional dimension for timestep. features here is the output of cnn, return from the CNN forward. 
        # embeddings is some numbers/index of words/captions
        # features as the first word of lstm. 
        hiddens, _ = self.lstm(embeddings) # embeddings are features of images and captions.  
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    # For evaluation 
    def caption_image(self, image, vocabulary, max_length=50): 
        result_caption = []  # use predicted word for future word.
        # For evaluation, torch.no_grad() to inference without gradient calculation. 
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0) # unsqueeze(0) to add 1 additional dimension for the batch.
            states = None # initialize states, hidden_state and cell_state of lstm

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.unsqueeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]








