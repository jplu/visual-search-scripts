# visual-search-scripts
Scripts to prepare a visual search process.

## Install requirements
Run:
```
pip install -r requirements.txt
```

And install [FAISS](https://github.com/facebookresearch/faiss)

## Download the Amazon image dataset
Before to download the dataset, you have to request an access to the owner of the dataset to have the [download link](http://jmcauley.ucsd.edu/data/amazon/). Once you have the download link put it in the `download_amazon_dataset.py` script, and then run the following command line:
```
python download_amazon_dataset.py
```

## Create the embedding generator model
To create the embedding generator model, run the following command line:
```
python create_saved_model.py
```

### Create the Docker image
If you want to run your Docker image over a GPU, you have to install [Nvidia for Docker](https://github.com/NVIDIA/nvidia-docker) and run this command line:
```
docker run -d --gpus all --name serving_base tensorflow/serving:latest-gpu
```

Otherwise run this one:
```
docker run -d --name serving_base tensorflow/serving:latest
```

Then, run these commands:
```
mkdir model
mv resnet18 model
mkdir model/resnet18/1
mv model/resnet18/variables model/resnet18/saved_model.pb model/resnet18/1
docker cp model/resnet18 serving_base:/models/resnet18
docker commit --change "ENV MODEL_NAME resnet18" \
    --change "ENV PATH $PATH:/usr/local/nvidia/bin" \ 
    --change "ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64" serving_base <image-name>
docker kill serving_base
docker rm serving_base
```

## Create the FAISS index
Run:
```
python create_faiss_index.py
```

## Run a visual search
To run a visual search over your images and index, you can use the corresponding [backend](https://github.com/jplu/visual-search-backend) and [frontend](https://github.com/jplu/visual-search-frontend).
