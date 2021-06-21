# Music-genre-classifier

Implementing different machine learning models on input music data and finding the most efficient method.

We have compared KNN and other ensemble features on GTZAN [Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

*model.py

We Scale our features using scaler()\
Training of model on logisitic regression\
We select main 30 audio features out of the given 55 using feature permutance\
We now train those models on various\
Hyperparameter tuning is performed\
Those trained weights are saved in pickle files

*app.py

Flask is imported\
The program extracts input from user in index.html\
The input saved in file extracts audio data using librosa library\
Feature extraction takes place (the same important 30 features are extracted)\
Refer the [official librosa docs](https://librosa.org/doc/latest/feature.html) for this\
We load the pickle files of trained models and run on the data\
The input is splitted into 10 intervals of 3 seconds out of 30 seconds audio as to compare to the dataset where similar has been done\
It is broken to create a much bigger dataset from much lesser data\
We test the data input data and send the results back to index.html

*index.html

Front-end for the project which extracts the audio file for user\
plays it\
Returns the recognized genre and model used along\

You can find the hosted website here: [Click to View](https://music-genre-classifier.sparshnagpal.repl.co/)

Check my published paper on this topic [Music Genre Classification Using Machine Learning Models](https://doi.org/10.22214/ijraset.2021.35381)

![Input](https://github.com/0sparsh2/Music-genre-classifier/blob/main/input.png)
![Output](https://github.com/0sparsh2/Music-genre-classifier/blob/main/output.png)


