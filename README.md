# Arceus

Hi everyone! Welcome to our distributed training framework, built from first principles leveraging model parallelism to optimize training on Apple M-series clusters. Arceus now supports model parallelism across multiple devices on the same local network!

## Run
To run this version of Arceus, please first initialize a virtual environment and install all required modules
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run Arceus across multiple machines on your local network, first initialize the Flask server to handle device registration and training orchestration
```shell
python -m api.server
```

## Training Options

### 1. Distributed Neural Network Training

First create register training job through the API, outlining the model and dataset configurations

```shell
curl -X POST http://localhost:4000/api/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "model_config": {
         "layer_sizes": [784, 128, 64, 10]
       },
       "dataset_config": {
         "name": "MNIST",
         "batch_size": 256,
         "val_batch_size": 1000,
         "normalize": {
           "mean": [0.1307],
           "std": [0.3081]
         }
       }
     }'
```
This returns a response with a job ID.

Now, register your devices. There must be atleast 1 and at most the number equal to the layers of your model.
On each of your devices, run
```shell
python -m nn.device_client --api-url <flask-server-ip> --job-id <job-id>
```

Then, initialize your devices with the layers of the model by specifying the job
```shell
curl -X POST http://<flask-server-ip>/api/network/initialize/<job-id>
```

Finally, train with the following request
```shell
curl -X POST http://<flask-server-ip>/api/network/train/<job-id> \
     -H "Content-Type: application/json" \
     -d '{"epochs": 10, "learning_rate": 0.1}'
```

### 2. Transformer Model Training

To train a transformer model (GPT), first register a training job with the transformer configuration:

```shell
curl -X POST http://localhost:4000/api/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "model_config": {
         "type": "transformer"
       },
       "dataset_config": {
         "source": "gpt/data.txt"
       }
     }'
```

Then start training directly (no device registration or initialization needed):
```shell
curl -X POST http://<flask-server-ip>/api/network/train/<job-id> \
     -H "Content-Type: application/json" \
     -d '{}'
```

The transformer model will use the hyperparameters defined in `gpt/model.py` and train on the data from the specified source file.

## Common Messages

You will see a message on your server when training starts:
```
127.0.0.1 - - [26/Nov/2024 10:40:56] "POST /api/network/train HTTP/1.1" 200 -

Starting distributed training across devices...
Learning rate: 0.1
Quantization bits: 8
Device: mps
```

If that didn't work and you got an SSL Certificate error, run this then try again
```
cd /Applications/Python\ 3.11/
./Install\ Certificates.command
```
