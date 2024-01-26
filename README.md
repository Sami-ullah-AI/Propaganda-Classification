# Propaganda-Classification
Propaganda is a powerful tool used by governments, organizations, and individuals to influence 
public opinion and behavior. As the spread of information becomes more widespread through the 
internet and social media, the need to identify and combat propaganda has become increasingly 
important. In this assignment, we will explore different approaches to classify whether a sentence 
contains propaganda or not, as well as techniques to identify the propaganda technique used in a 
given text snippet or span.
We will investigate four different approaches to classification: text probability based on n-gram 
language models, text similarity or classification based on uncontextualised word embedding 
methods such as word2vec, neural language models, and pretrained large language models such 
as BERT. For each approach, we will describe the methodology, any hyper-parameters used, and 
evaluate their performance in classifying propaganda.
The findings from this assignment will help us understand the strengths and weaknesses of 
different techniques in identifying propaganda and can potentially be used to improve existing 
algorithms for detecting and combating propaganda in various domains.
In this assignment, we will use four different models to classify whether a sentence contains 
propaganda or not, as well as techniques to identify the propaganda technique used in a given text 
snippet or span.
The first approach we will use is based on n-gram language models. This approach uses 
probabilities of n-grams, or sequences of n words, to determine whether a sentence contains 
propaganda. We will experiment with different values of n and smoothing techniques to optimize 
the model's performance.
The second approach we will use is based on uncontextualised word embedding methods, 
specifically the word2vec algorithm. This method represents words as vectors in a highdimensional space, where words with similar meanings are closer together. We will use this model 
to determine the similarity between sentences and classify them as containing propaganda or not.
The third approach we will use is based on neural language models. This approach involves 
training a neural network to predict the next word in a sentence, given the previous words. We will 
experiment with different architectures and hyperparameters to optimize the model's performance.
The fourth approach we will use is based on pretrained large language models, specifically the 
BERT model. BERT is a transformer-based language model that has been trained on a large corpus 
of text data and has achieved state-of-the-art results on various natural language processing tasks. 
We will fine-tune the BERT model on our propaganda classification task and evaluate its 
performance.
Each model has its strengths and weaknesses, and we will compare their performance and practical 
considerations in this assignment. By understanding the capabilities and limitations of each model, 
we can choose the most appropriate approach for a given propaganda detection task.
Methadology
In this section, we will describe the methodology used for each approach in classifying propaganda 
and identifying propaganda techniques.
N-gram language model:
For this approach, we first preprocess the dataset by converting all text to lowercase and removing 
any punctuation marks. We then split the sentences into n-grams, where we experiment with 
different values of n ranging from 1 to 5. We then calculate the probabilities of each n-gram 
occurring in the dataset and use this information to calculate the probability of a sentence 
containing propaganda using the Bayes' theorem.
To avoid zero probabilities, we use smoothing techniques such as Laplace smoothing or GoodTuring smoothing. We then set a threshold probability above which we classify a sentence as 
containing propaganda. We will evaluate the model's performance using metrics such as accuracy, 
precision, recall, and F1-score.
Word2vec:
For this approach, we first tokenize the sentences and remove any stop words. We then train the 
word2vec model on the dataset to learn word embeddings that represent each word as a vector in 
a high-dimensional space. We then calculate the cosine similarity between the sentence vectors 
and a reference vector that represents the concept of propaganda.
We set a threshold similarity above which we classify a sentence as containing propaganda. We 
will evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-
score.
Neural language model:
For this approach, we first preprocess the dataset by converting all text to lowercase and removing 
any punctuation marks. We then tokenize the sentences and create sequences of tokens with a fixed 
length, which will be used as input to the neural network. We then split the dataset into training 
and validation sets.
We then build a neural network architecture that consists of an embedding layer, a recurrent layer 
such as LSTM or GRU, and a dense layer with a sigmoid activation function to output a binary 
classification. We then compile the model using binary cross-entropy loss and train it on the 
training set. We will experiment with different architectures, hyperparameters such as learning 
rate, dropout rate, and sequence length to optimize the model's performance.
We will evaluate the model's performance using metrics such as accuracy, precision, recall, and 
F1-score on the validation set.
BERT:
For this approach, we first preprocess the dataset by converting all text to lowercase and removing 
any punctuation marks. We then tokenize the sentences and convert them into input features that 
BERT can process. We then split the dataset into training and validation sets.
We use the pre-trained BERT model and fine-tune it on our propaganda classification task by 
training it on the training set and evaluating its performance on the validation set. We experiment 
with different hyperparameters such as learning rate, number of epochs, and batch size to optimize 
the model's performance.
We will evaluate the model's performance using metrics such as accuracy, precision, recall, and 
F1-score on the validation set.
Overall, we will compare the performance of each approach using the metrics mentioned above, 
as well as practical considerations such as training time, model complexity, and scalability. By 
evaluating each approach's strengths and weaknesses, we can determine the most suitable approach 
for a given propaganda detection task.
