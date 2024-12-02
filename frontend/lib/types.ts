export type Device = {
  device_id: number;
  total_teraflops: number;
  chip: string;
  device_layers: {
    [key: string]: number[];
  };
};

export type TimingData = {
  avg_backward: number;
  // avg_backward_tflop: number;
  avg_comm: number;
  avg_forward: number;
  // avg_forward_tflops: number;
  avg_prep: number;
  avg_update: number;
  batch_idx: number;
  // total_computation: number;
  // total_overhead: number;
  device_data: Device[];
};

export type EpochStats = {
  epoch: number;
  epochs: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  epoch_time: number;
};

export type TrainingData = {
  epoch: number;
  epochs: number;
  train_loss: number;
  train_acc: number;
  batch_idx: number;
  batch_time: number;
  num_batches: number;
  tokens_trained: number;
  total_tokens: number;
};
