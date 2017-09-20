In this project, we implemented a caption generator. It was done following the idea described in https://arxiv.org/pdf/1411.4555v1.pdf

The images were first processed through Alexnet (using an implementation found on http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/). We then used the results processed by the 7th layer of this network as input to the caption generator, instead of the raw images. -> the codes for this are in the traitementImgs folder
For the captions, we used a vector representation of words, using a word2vec model.

We then used a simple LSTM, that we trained on the Flickr8k dataset.

Some examples of the result can be found in the results.pdf file.
Some statistics on them : on 60 images randomly (the first 60 ones...) selected from the tests set, the captions generated :
 - made no sense for 32 of them
 - were somehow related to elements in the picture for 22 of them
 - actually provided a rather good description of the picture for 6 of them

(Looking back on it, a more integrated pipeline would definitely have made the choice and testing of hyperparameters easier, and would probaly have yielded better results)