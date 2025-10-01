
Fake New Detection
# Pipeline of this project:

1.Problem understanding

2.data colllection

3.data explore

4.data preprocessing

5.tokenizer(use transformer)

6.load the pretrained model

7.fine tune the pretrained model

8.to tain the fine tune the pretrained model using Trainer class

9.Evaluate the model

10.Save the Model

11.deploy

#  Step 1:Problem Understanding

Problem=> classify weather given new article is fake or real

input=>text(title,content,both(title,content))

output=>Binary value(Fake/real)

# Step 2: Data collection

we are using the dataset

```
ISOT FAKE NEW DATASET
```
the ISOT Fake News Dataset contains two separate CSV files:

True.csv → Contains real news articles.

Columns: title, text

Every row represents a real news article.

Fake.csv → Contains fake news articles.

Columns: title, text

Every row represents a fake news article.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the datset
df_real=pd.read_csv("/content/True.csv")
df_fake=pd.read_csv("/content/Fake.csv")

df_real.head(4)

df_fake.head(5)

"""Subject / Date / Source / Author:

These are not necessary for text-based classification.

You can safely drop them before training.

They may introduce noise or bias if the model learns patterns like “all news from a certain date/source = real/fake.”
"""

# we drop the subject and date column .we are not effect the training and inference(prediction).
df_real = df_real[["title", "text"]]
df_fake = df_fake[["title", "text"]]

df_real.head(2)

"""merge the dataset into one single csv file"""

# Add labels: 1 = real, 0 = fake
df_real["label"] = 1
df_fake["label"] = 0

df_real.head(2)

df_fake.head(2)

# Combine & shuffle
df = pd.concat([df_real, df_fake], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.head()

df.sample(5)

df['content'] = df['title'] + " " + df['text']
df = df[['content', 'label']]

df.head(2)

"""# Step 3:Data Explore"""

df.shape

"""these dataset has 44898 row and 2 column"""

df.info()

"""check the null value"""

df.isnull().sum()

df = df.dropna()  # drop rows with NaN
df = df[df['content'].str.strip() != ""]  # remove empty strings

df.head(2)

"""check the class distribution"""

print(df['label'].value_counts())

"""this is not imblanced data

now its plot the data
"""

df['label'].value_counts().plot(kind='bar', title="Class Distribution")
plt.show()

df['label'].value_counts().plot(kind='pie', title="Class Distribution")
plt.show()

"""now this is clear see class are perfect balanced

now check the text length distribution

for text column
"""

df.head(1)

print("Sample fake news:\n", df[df['label']==0].sample(1)['content'].values[0])
print("Sample real news:\n", df[df['label']==1].sample(1)['content'].values[0])

"""# Step 4: Data Preprocessing

Lowercasing

Removing punctuation, special characters, numbers

Removing extra spaces

Optional: remove stopwords

convert the whole dataset in lowecase because we are using bert-base-uncased
"""

def lower(text):
  text=text.lower()
  return text

df['content'] = df['content'].apply(lower)

df['content'].head(2)

"""removing punction,special character ,numbers"""

import re
import string
exclude=string.punctuation
def clean_text(text):
  text=re.sub(r'\d+', '', text)
  text=text.translate(str.maketrans('','',exclude))
  return text

df['content'] = df['content'].apply(clean_text)

df['content'].sample(2)

"""Removing the white space"""

import re
def remove_space(text):
  text = re.sub(r'\s+', ' ', text).strip()
  return text

df['content'] = df['content'].apply(remove_space)

df['content'].sample(4)

from sklearn.model_selection import train_test_split
train_text,test_texts,train_labels,test_labels = train_test_split(
    df['content'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

"""# Step 5:Tokenizer using transformer and load the model BERT

install the transformer
"""

!pip install transformers

from transformers import AutoTokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize train and test
train_encodings = tokenizer(
    train_text.tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="tf"
)

test_encodings = tokenizer(
    test_texts.tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="tf"
)

"""Autotokenizer=> convert to raw text in token ids
AutoModelForSequenceClassification=> Load the pretrained model BERT

to convert the tensorflow dataset

tensorflow dataset=>raw tensor dataset to convert the object
"""

import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids":train_encodings['input_ids'],"attention_mask":train_encodings['attention_mask']},
    train_labels)).shuffle(10000).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids":test_encodings['input_ids'],"attention_mask":test_encodings['attention_mask']},
    test_labels)).batch(16)

# load the bert model
from transformers import TFAutoModel
model = TFAutoModel.from_pretrained(model_name,num_labels=2, use_safetensors=False )

"""freeze all layer"""

model.trainable = False
#input
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
# base model output
embedding = model(input_ids, attention_mask=attention_mask)[0]
# take[cls token]
cls_token = embedding[:, 0, :]
# classifier layer
output = tf.keras.layers.Dense(2, activation="softmax")(cls_token)
# final model
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

"""compile the model"""

from tensorflow.keras.losses import SparseCategoricalCrossentropy

optimizer = 'adamw'
loss = SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

import tensorflow as tf
print(tf.__version__)

"""fine tune the model"""

# # history = model.fit(
# #     train_dataset.shuffle(1000).batch(32),
# #     validation_data=test_dataset.batch(32),
# #     epochs=3
# # )
# history = model.fit(
#     train_dataset.shuffle(1000).batch(32),
#     validation_data=test_dataset.batch(32),
#     epochs=3
# )
history=model.fit(train_dataset,validation_data=test_dataset,epochs=3)

"""# Improve the accuracy and reduce the losss
using technique like:

1.Early stopping

2.To train the more  epochs



"""

from tf_keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True,verbose=1)

history=model.fit(train_dataset,validation_data=test_dataset,epochs=5,callbacks=[early_stopping])

"""Evaluate the model"""

from sklearn.metrics import accuracy_score
loss, accuracy = model.evaluate(test_dataset)
print(f"Test loss: {loss}, Test accuracy: {accuracy*100 : 2f}%")

"""now plot the loss and accuracy then clarify the model is overfitting or not"""

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""he model is performing well on both training and validation data, suggesting it is neither significantly overfitting nor underfitting.

Prediction
"""

def predict_fake_news(text):
    cleaned_text = remove_space(clean_text(lower(text)))
    encoding = tokenizer(
        [cleaned_text],
        padding="max_length",   #  force padding
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )
    prediction = model.predict(
        {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}
    )
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

    return "Real News" if predicted_class == 1 else "Fake News"

sample_text_1 = "The government has announced new economic reforms today."
sample_text_2 = "Aliens landed in New York City yesterday and took over Times Square."

print("Prediction for sample 1:", predict_fake_news(sample_text_1))
print("Prediction for sample 2:", predict_fake_news(sample_text_2))

"""now test the differnt text"""

examples = [
    "Breaking: Scientists discover a new planet that could support human life.",
    "The local mayor inaugurated a new hospital in the downtown area today.",
    "A celebrity claims to have traveled back in time using a secret machine.",
    "The stock market saw a significant rise after the central bank reduced interest rates.",
    "Reports suggest that chocolate cures all diseases if eaten daily."
]

for i, text in enumerate(examples, 1):
    print(f"Prediction for example {i}: {predict_fake_news(text)}")

"""Now sav the model for deployment"""

model.save("fake_news_model.h5")

# Save tokenizer
tokenizer.save_pretrained("fake_news_model_tokenizer")

# Save tokenizer
# tokenizer.save_pretrained("fake_news_model_tokenizer1.zip")

