import { Beer } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
        <ReactMarkdown
          className={cn(
            "prose prose-sm max-w-none",
            "prose-invert",
            "prose-p:leading-relaxed prose-pre:bg-stone-800 prose-pre:border prose-pre:border-stone-700",
            "prose-a:text-amber-400 prose-a:no-underline hover:prose-a:underline hover:prose-a:text-amber-300",
            "prose-strong:text-amber-200 prose-headings:text-amber-100",
          )}
          remarkPlugins={[remarkGfm]}
        >
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
