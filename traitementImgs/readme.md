Most of the code in calcImages.py was taken from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

This code first generates from the Flickr8k captions a list of the images name.
Then for each of those names, reads the image, sends it through Alexnet, and prints the output in a text file with the following format : 
nameOfTheImage#x y z t ...
Where x, y, z, t... are the output numbers of the 7th layer of the Alexnet neural network.

Due to computationnal limitations (this was run on a personnal computer), this code only treats 2 000 images at a time (those whose id is between the constants deb and fin defined l.48.)