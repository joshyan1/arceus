import { getData } from "@/components/models/data";
import Dashboard from "@/components/dashboard";
import { notFound } from "next/navigation";

export async function generateStaticParams() {
  const models = getData();
  return models.map((model) => ({ id: model.id }));
}

export default function ModelDashboard({ params }: { params: { id: string } }) {
  const model = getData().find((model) => model.id === params.id);
  if (!model) {
    notFound();
  }
  return <Dashboard model={model} />;
}
