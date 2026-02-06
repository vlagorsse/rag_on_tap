import { Beer } from "lucide-react";

export function Header() {
  return (
    <header className="h-16 border-b border-stone-800 flex items-center justify-between px-6 bg-stone-950/50 backdrop-blur-sm z-10">
      <div className="flex items-center gap-2 md:hidden">
        <Beer className="text-amber-500 h-6 w-6" />
        <span className="font-bold">Beer RAG</span>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-xs py-1 px-2 bg-amber-500/10 text-amber-500 rounded-full border border-amber-500/20 font-medium">
          Gemini 2.5 Flash Lite
        </span>
      </div>
    </header>
  );
}
