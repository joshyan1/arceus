import { getData } from "@/components/models/data";
import Dashboard from "@/components/dashboard";

export async function generateStaticParams() {
  const models = getData();
  return models.map((model) => ({ id: model.id }));
}

export default function ModelDashboard({ params }: { params: { id: string } }) {
  return <Dashboard id={params.id} />;
}
