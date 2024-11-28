# API

## Run
To start your Flask Server, run
```shell
python -m api.server
```

## Endpoints
- [Create a Job](#create-job)
- [List Jobs](#list-jobs)
- [Retrieve Job Details](#get-job-details)
- [Register Device](#register-device)
- [Initialize Network](#initialize-neural-network)
- [Start Trainig](#train-neural-network)
- [Unregister Device](#unregister-device)


### Create Job
Creates a new training job with specified model and dataset configurations
```
POST /api/jobs
```

Example:
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

Response:
```json
{
    "message": "Job created successfully",
    "job_id": "1"
}
```

### List Jobs
Retrieves all active jobs
```
GET /api/jobs
```

Example:
```shell
curl http://localhost:4000/api/jobs
```

Response:
```json
{
    "jobs": [
        {
            "job_id": "1",
            "status": "created",
            "devices": 0,
            "model_config": {
                "layer_sizes": [784, 128, 64, 10]
            },
            "dataset_config": {
                "name": "MNIST",
                "batch_size": 256
            }
        }
    ]
}
```

### Get Job Details
Get details of a specific job
```
GET /api/jobs/<job_id>
```

Example:
```shell
curl http://localhost:4000/api/jobs/1
```

Response:
```json
{
    "job_id": "1",
    "status": "created",
    "devices": 0,
    "model_config": {
        "layer_sizes": [784, 128, 64, 10]
    },
    "dataset_config": {
        "name": "MNIST",
        "batch_size": 256
    }
}
```

### Register Device
Registers a new device with a specific job
```
POST /api/devices/register
```

Example:
```shell
curl -X POST http://localhost:4000/api/devices/register \
     -H "Content-Type: application/json" \
     -d '{
       "ip": "192.168.1.100",
       "port": 5001,
       "job_id": "1"
     }'
```

Response:
```json
{
    "message": "Device registered successfully",
    "device_id": 1
}
```

### Initialize Neural Network
Initializes neural network by allocating layers to registered devices for a specific job
```
POST /api/network/initialize/<job_id>
```

Example:
```shell
curl -X POST http://localhost:4000/api/network/initialize/1
```

Response:
```json
{
    "message": "Network initialized successfully"
}
```

### Train Neural Network
Begin training of neural network for a specific job
```
POST /api/network/train/<job_id>
```

Example:
```shell
curl -X POST http://localhost:4000/api/network/train/1 \
     -H "Content-Type: application/json" \
     -d '{
       "epochs": 10,
       "learning_rate": 0.1
     }'
```

Response:
```json
{
    "message": "Training started",
    "epochs": 10,
    "learning_rate": 0.1
}
```

### Unregister Device
Remove a registered device
```
DELETE /api/devices/<port>
```

Example:
```shell
curl -X DELETE http://localhost:4000/api/devices/5001
```

Response:
```json
{
    "message": "Device unregistered successfully"
}
```

## Error Responses
All endpoints may return error responses in the following format:
```json
{
    "error": "Error message description"
}
```

Common HTTP status codes:
- 400: Bad Request (missing or invalid parameters)
- 404: Not Found (invalid job_id or device not found)
- 409: Conflict (device already registered)
- 500: Internal Server Error

