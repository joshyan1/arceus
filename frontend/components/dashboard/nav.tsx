import Link from "next/link";

export default function Nav() {
  return (
    <nav className="flex h-12 w-full shrink-0 select-none items-center justify-between border-b px-4">
      <div className="flex items-center gap-2 text-lg">
        <Link href="/">
          <div className="font-supply transition-all hover:text-muted-foreground">
            ARCEUS
          </div>
        </Link>
        <div className="translate-y-px text-2xl">/</div>
        <div className="translate-y-px">Some Model</div>
      </div>

      <div className="size-8 rounded-full bg-gradient-to-br from-primary to-primary/25"></div>
    </nav>
  );
}
