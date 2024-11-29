export type Device = {
  id: number;
  name: string;
  cpu: string;
  tflops: number;
  task: number[];
  usage: number;
  battery: number;
};
