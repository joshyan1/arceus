import Link from "next/link";
import { Button } from "../ui/button";
import { Settings } from "lucide-react";

export default function Nav() {
  return (
    <nav className="relative flex h-14 w-full shrink-0 select-none items-center justify-between border-b px-4">
      <div className="flex items-center gap-2 text-lg">
        <Link href="/">
          <div className="font-supply transition-all hover:text-muted-foreground">
            ARCEUS
          </div>
        </Link>
        <div className="translate-y-px text-2xl">/</div>
        <div className="translate-y-px">Some Model</div>
      </div>

      <Button variant="ghost" size="icon">
        <Settings className="size-4" />
      </Button>
    </nav>
  );
}
