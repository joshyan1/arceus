import Transformer1 from "@/assets/images/transformer1.png";
import Transformer2 from "@/assets/images/transformer2.png";
import Transformer3 from "@/assets/images/transformer3.png";
import NeuralNetwork1 from "@/assets/images/nn1.png";
import NeuralNetwork2 from "@/assets/images/nn2.png";
import NeuralNetwork3 from "@/assets/images/nn3.png";

const transformerImages = [Transformer1, Transformer2, Transformer3];
const neuralNetworkImages = [NeuralNetwork1, NeuralNetwork2, NeuralNetwork3];

export default function getModelImage(modelType: string) {
  const images =
    modelType === "transformer" ? transformerImages : neuralNetworkImages;
  const randomImage = images[Math.floor(Math.random() * images.length)];
  return randomImage;
}
