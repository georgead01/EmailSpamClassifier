# EmailSpamClassifier
A spam email classifier using ML and NLP.

# script.py

The script could be divided into two parts:
1. Preparing the data.
2. Training the model.

## Data Preparation

The data is prepared in three main steps:
1. Import the data and extract features.
2. Split into training and testing set.
3. Tokenize data.

## Training Model

The model we used is of the following architecture:
![model_graph](https://github.com/georgead01/EmailSpamClassifier/assets/23529317/a57fe30d-5d88-4a43-af60-40fb08c2679f)

Note that:
1. Our embedding layer serves the purpose of reducing the dimensions of our input.
2. Our output layer is of size 1 with a sigmoid activation function --> binary classification. For an email classifier with multiple categories, we would use an output layer of the same size as the number of categories and a softmax activation function.

We train our model using binary crossentropy loss (again, binary classification), and we're able to get ~96% accuracy on our testing data.

<img width="790" alt="Screenshot 2023-06-23 at 1 41 30 PM" src="https://github.com/georgead01/EmailSpamClassifier/assets/23529317/645b62f6-3365-4855-8151-1ed0edf18a44">

# spam_ham_dataset.csv

The data we used to train the model could be found on Kaggle: https://www.kaggle.com/datasets/venky73/spam-mails-dataset

We used the data from the 'text' column as our input, and the data from the 'label_num' column as our labels.
