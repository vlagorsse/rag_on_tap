import { Beer, Info, RefreshCw, Github } from "lucide-react";

export function Sidebar() {
  return (
    <div className="w-64 bg-stone-900 border-r border-stone-800 flex flex-col hidden md:flex">
      <div className="p-4 flex items-center gap-3 border-b border-stone-800">
        <div className="bg-amber-500 p-2 rounded-lg">
          <Beer className="text-stone-950 h-6 w-6" />
        </div>
        <h1 className="font-bold text-lg tracking-tight">
          Beer RAG on steroids !
        </h1>
      </div>

      <div className="flex-1 p-4 overflow-y-auto">
        <button className="w-full flex items-center gap-2 bg-stone-800 hover:bg-stone-700 p-3 rounded-md transition-colors text-sm mb-4">
          <RefreshCw className="h-4 w-4" />
          New Session
        </button>

        <div className="space-y-1">
          <p className="text-xs font-semibold text-stone-500 uppercase tracking-wider mb-2 px-1">
            Recent Queries
          </p>
          <div className="text-sm text-stone-400 p-2 hover:bg-stone-800 rounded cursor-pointer truncate">
            Hoppy IPAs with Citra
          </div>
          <div className="text-sm text-stone-400 p-2 hover:bg-stone-800 rounded cursor-pointer truncate">
            Stout brewing techniques
          </div>
          <div className="text-sm text-stone-400 p-2 hover:bg-stone-800 rounded cursor-pointer truncate">
            Belgian Yeast profiles
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-stone-800 space-y-4">
        <div className="flex items-center gap-2 text-xs text-stone-500 hover:text-stone-300 cursor-pointer">
          <Info className="h-4 w-4" />
          About HyPE RAG
        </div>
        <div className="flex items-center gap-2 text-xs text-stone-500 hover:text-stone-300 cursor-pointer">
          <Github className="h-4 w-4" />
          View Source
        </div>
      </div>
    </div>
  );
}
