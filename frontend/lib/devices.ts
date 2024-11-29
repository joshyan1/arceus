export type Device = {
  id: number;
  name: string;
  cpu: string;
  tflops: number;
  task: number[];
  usage: number;
  battery: number;
};

export const you: Device = {
  id: 1,
  name: "CURSOR REJECT",
  cpu: "M1",
  tflops: 1.2,
  task: [3],
  usage: 0.5,
  battery: 1,
};

export const devices: Device[] = [
  {
    id: 2,
    name: "THE RAGE",
    cpu: "M3 MAX",
    tflops: 3.8,
    task: [1],
    usage: 0.7,
    battery: 1,
  },
  {
    id: 3,
    name: "PLAYSTATION 5",
    cpu: "M2",
    tflops: 2.9,
    task: [2],
    usage: 0.3,
    battery: 0.5,
  },
];
