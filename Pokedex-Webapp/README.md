# Pokemon Classifier Webapp

An ML pokemon classifier web application because I don't know enough about pokemon to classify them myself.

## Prerequisites

Install necessary python packages:

```
pip install -r requirements.txt
```

## Train Model

To train the model: 

1) Download and unzip the dataset into the **model** folder - [https://techx.blob.core.windows.net/pokemon/dataset.zip](https://techx.blob.core.windows.net/pokemon/dataset.zip)
2) Run the **train.py** file with the following parameters in the **model** folder:

```
python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle
```

## Test Model

To test the model:

1) Download and the weights into the **model** folder (if not previously trained) - [https://techx.blob.core.windows.net/pokemon/pokedex.model](https://techx.blob.core.windows.net/pokemon/pokedex.model)
2) Run the **classify.py** file with the following parameters in the **model** folder:

```
python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png
```

## Run The Web Application

To run the web application:

1) Download and the weights into the **model** folder (if not previously trained) - [https://techx.blob.core.windows.net/pokemon/pokedex.model](https://techx.blob.core.windows.net/pokemon/pokedex.model)
2) Run the **app.py** file from the **root** directory:

```
python app.py
```

3) Go to **localhost:5000** to access the application from the browser of your choice.
