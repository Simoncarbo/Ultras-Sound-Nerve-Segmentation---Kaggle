#My first deeplearning project

This project was built for the Kaggle Ultrasound Nerve Segmentation competition. 
Basic information about data and context can be found on their website: https://www.kaggle.com/c/ultrasound-nerve-segmentation.
The solution presented here ranked 57th on 923 with a dice score of 0.691 (first place got 0.732). 
Some original ideas have been coded during the project, and I would be glad if it helps anyone.

##Key ideas
###Architecture
The solution uses a deep convolutional network, that has been adapted for segmentation such that image-level features 
can be learned for the classification of each pixel. While most of the participants used the U-net, my architecture is mostly inspired by the Hypercolumns model described in https://arxiv.org/abs/1411.5752. Since the images had more or less constant spatial structure (nerves mostly at same location), locally connected layers were used in parallel to convolutional ones from the 10x14 resolution. I also used SemiShared layers, a novel layer that shares filter weights only in blocks (with specified size) of the image. In a certain way, it provides a compromise between convolutional and locally connected layers. To help the coding, SemiShared layer only works for 1x1 filters. Finally, the model also emits a unique scalar value, that is multiplied with the final mask inside the network. This was done to help the network managing labelling errors: very similar images where sometimes labelled with and without nerve.

###Cost function
For learning, the dice coefficient was used with two slight modifications:
- a smoothing factor in the denominator
- a factor that multiplies training examples with nerve, since nerve presence ratio varied a lot in train/validation/test sets

The second modification could increase validation score from 0.7 to 0.72 even for a factor close to one (1.085).

###Subject information
The training data contains images taken from 47 subjects, 120 images each. Since data contains much more variation 
between subjects than in a single subject, every batch was composed of exactly one image per subject. This allows for 
batches that are representative of the complete training set. This results in more stable training (better convergence) 
and should improve batch normalization as well. Finally, the validation split was made subject-wise for better performance evaluation.

### Ensembling on epochs
During training, while training loss was increasing, validation score varied a lot (0.69-0.72) without noticably decreasing.
I couldn't figure out why and finally, ensembling over the best epochs gave the best results.

###Post-processing
A small post-processing is performed on the results. Basicly smoothing and deleting small masks. The gain was ~0.004 in public and
private test sets. Other participants seem to have post-processing methods resulting in ~0.02 gain based on the same ideas but 
with optimized parameters. I don't know if such results could be transposed to this model.

##Possible improvement:
- Explore more network architectures
- Data augmentation (horizontal flip was helpful apparently, elastic deformation wasn't)
- Optimized post-processing

##Config
The code runs on Python 3.4 and uses the Keras library with Theano (0.9.0) backend. Since I slightly changed Keras code 
for new type of layers, the code only runs with my Keras fork. You can download it on my github account.
You will also need at least 6GB of RAM since we are working with high resolution images (I was using 12GB).

##Conclusion and acknowledgements
After months of reading, this is my first deeplearning project. Deeplearning is supposedly a lot of trial and error, 
but I tried to focus on the thinking part. While it couldn't bring me to the top submissions, I learned a lot and tried out some original ideas.
Of course, all this wouldn't have been possible without all the work that has been put in Keras and Theano. I'll never
thank the contributors enough. I also want to thank Marko Jocic for his starter code for the competition.
I did not evaluate the gain from the presented ideas, but will potentially publish some of them in a scientific paper. If you have any questions or want to do some collaboration, don't hesitate to send me a message.
