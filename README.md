# BERT-based NN TopCoder Pricing Model

> This is the repo for Benjamin's research thesis

The previous repo containing other models are too messy to manage - That's what happened when you have no idea about a data analysis research

So instead of making the original repo messier, I created a new repo to singly focus on builing the pricing model using BERT and Neural Network. There are several parts to builde the pricing model:

1. Fine-tune bert with my requirement text
2. Customize network and append meta data features in the BERT last layer states
3. Train the model for a classification task

I will have some code/data copy-paste from the [previous repo](https://github.com/BenjiTheC/TopCoderDataAnalysis.git) to save my time.

---

## Insturctions of code struture

> This setion is for Mingyang to locate the code that needs his help to review
>
> Thanks for your help in advance Mingyang :D

- `get_data.py`: The data of TopCoder challenges is fetched and stored in a MySQL database on the server of Stevens, this script is for fetching the original data.

- `tc_data.py`: This file define a class `TopCoder` that read the data into `pandas.DataFrame` with some simple preprocessing, including manuanly filtering out the data used for training of the neural network. It will return the text and metadata for 4,800-ish challenge data.

- `model_tcpm_distilbert.py`: **[HELP NEEDED HERE!]** This file implement the neural network model I want to train in two ways:
    1. Using [functional api](https://www.tensorflow.org/guide/keras/functional) of `Tensorflow`. The model returned should be trainable as standard `tf.keras.Model` using `model.fit` method.
    2. Implement `TCPMDistilBertClassification`, a subclass of `TFDistilBertPreTrained`, which essentially is a subclass of `tf.keras.Model`. And it can be trained using `TFTrainer` from the `transformers`.

- `fine_tune_bert.py`: **[HELP NEEDED HERE AS WELL]** This file implement the function `build_dataset` that convert the data from `pandas.DataFrame` to `tf.data.Dataset`. And a `fine_with_tftrainer` which insantiates the `TCPMDistilBertClassification` and trains the model.

- `preprocessing_util.py`: Some text preproccessing utility funtion.
