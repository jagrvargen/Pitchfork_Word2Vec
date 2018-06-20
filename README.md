# A Simple Word2Vec Model in TensorFlow

This project is a first attempt to build a skip-gram Word2Vec model in Python 3.6 and TensorFlow 1.8 trained on album reviews from the popular contemporary music review site Pitchfork. Using 18393 reviews sourced from Kaggle.com, I trained a neural network using Noise Contrastive Estimation to measure loss and optimized using the Adam Optimizer.

## Getting Started

### Environment 
All packages were installed in the Anaconda 3.5.4 package management environment. Libraries used:
TensorFlow 1.8
Natural Language ToolKit 3.3

## Deployment

This model was trained using FloydHub cloud services in order to have access to a GPU. It is possible to train the model on a CPU, but the process may take several hours. In order to begin training the model, simply type the command:
python3 clean_text.py. All the functionality to parse and batch the data is included in this file. Just make sure that the reviews\_corpus.txt file exists in the same directory.

## Author
Jesse Hedden <jesse.hedden@holbertonschool.com>

## Acknowlegdments

- This project would not have been possible without constant reference to the excellent online tutorials on adventuresinmachinelearning.com. In order to produce visual output and track loss accross training iterations, I referred to their code in the [repository](https://github.com/adventuresinML/adventures-in-ml-code/blob/master/tf%5Fword2vec.py)

- The [Introduction to Machine Learning Course by Andrew Ng](coursera.com), as well has his excellent tutorials on [RNNs](https://www.youtube.com/watch?v=5Vl-bK7tfD8&list=PLBAGcD3siRDittPwQDGIIAWkjz-RucAc7)
