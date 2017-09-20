TRAINING : 
The file training.py takes as input the training set and saves the neural network weights after training.
Input format : 
One line per couple (image, caption) : 
idImage#number of words in the caption, including the specific BEGIN and END words#the vector representing the image as calculated through the Alexnet (with ' ' as a separator between coordinates)#the vector representing the first word, with ' ' as a separator#idem for the second word...
Constants to change : dimMot = word dimension, dimImgs = images dimension, longMaxLeg = max length of the generated captions


TESTING : 
The file testing.py loads the weights saved by the training code. It takes as input the images, and a dictionnary (words <-> their vector representation)
Input format : 
(ImATester.txt) -> the images to create captions for
One line per image : idImg#the vector representing the image with ' ' as a separator between coordinates

(dico_lemma.txt) -> the dictionnary
One line per word : word#the vector representing the word with ' ' as a separator between coordinates

Output format : 
(res.txt)
For eahc image : 
idImage#word1 word2 ...

The testing code is extremely slow. Indeed, for each word vector generated, it runs through the whole dctionnary, to replace it with the closest known word. Some testing showed that doing this on the fly,  when generating the caption (and hence, using the replaced word to generate the next word) gave much better results than generating the whole caption and then "translating" using the closest word to the generated results.