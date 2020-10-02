
![Alt text](static/stanford_dogs.png)

The repository consists of a classification network based on pretrained Xception architercture, created by Francois Chollet and descibed in []. The repository contains a notebook, showing the user how to train a network using stanford-dog-classification dataset. Moreover, the repository shows how one can use Mlflow package to run experiments for testing different hyperparameters/architectures in for that task. Furthermore, Flask micro-framework has been used to provide a means of communication with the model to assess the prediction of the model via REST API.

You may also run Streamlit to check, how the model behaves, and provide your own images (just copy them to img folder) and provide them to the network to obtain accuracy.
