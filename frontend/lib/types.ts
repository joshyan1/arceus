export type Device = {
  id: number;
  name: string;
  cpu: string;
  tflops: number;
  task: number[];
  usage: number;
  battery: number;
};

export type TimingData = {
  avg_backward: number;
  avg_backward_tflop: number;
  avg_comm: number;
  avg_forward: number;
  avg_forward_tflops: number;
  avg_prep: number;
  avg_update: number;
  batch_idx: number;
  total_computation: number;
  total_overhead: number;
};

export type EpochStats = {
  epoch: number;
  epochs: number;
  val_loss: number;
};

export type TrainingData = {
  epoch: number;
  epochs: number;
  train_loss: number;
  train_acc: number;
  batch_idx: number;
  batch_time: number;
};
