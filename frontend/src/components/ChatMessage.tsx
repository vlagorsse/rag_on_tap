import { Beer } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { cn } from "../lib/utils";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isLoading?: boolean;
}

export function ChatMessage({ role, content, isLoading }: ChatMessageProps) {
  return (
    <div
      className={cn(
        "flex gap-4 group",
        role === "user" ? "justify-end" : "justify-start",
      )}
    >
      {role === "assistant" && (
        <div className="h-8 w-8 rounded-full bg-amber-500 flex items-center justify-center flex-shrink-0 mt-1">
          <Beer className="h-4 w-4 text-stone-950" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          role === "user"
            ? "bg-amber-600 text-white rounded-tr-none"
            : "bg-stone-900 border border-stone-800 text-stone-200 rounded-tl-none",
        )}
      >
        <ReactMarkdown className="prose prose-invert prose-sm max-w-none">
          {content}
        </ReactMarkdown>
        {isLoading && content === "" && (
          <div className="flex gap-1 py-2">
            <div
              className="h-1.5 w-1.5 bg-stone-500 rounded-full animate-bounce"
              style={{ animationDelay: "0ms" }}
            />
            <div
              className="h-1.5 w-1.5 bg-stone-500 rounded-full animate-bounce"
              style={{ animationDelay: "150ms" }}
            />
            <div
              className="h-1.5 w-1.5 bg-stone-500 rounded-full animate-bounce"
              style={{ animationDelay: "300ms" }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
