import { Send } from "lucide-react";

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
}

export function ChatInput({
  input,
  setInput,
  onSend,
  isLoading,
}: ChatInputProps) {
  return (
    <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-stone-950 via-stone-950 to-transparent">
      <div className="max-w-3xl mx-auto relative">
        <textarea
          className="w-full bg-stone-900 border border-stone-800 rounded-2xl pl-4 pr-12 py-4 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/50 resize-none shadow-2xl"
          placeholder="Ask about IPAs, stouts, brewing tips..."
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
        />
        <button
          onClick={onSend}
          disabled={!input.trim() || isLoading}
          className="absolute right-3 bottom-3 p-2 bg-amber-500 hover:bg-amber-400 disabled:bg-stone-800 disabled:text-stone-600 rounded-xl transition-all text-stone-950"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>
      <p className="text-[10px] text-center mt-2 text-stone-600">
        RAG-on-Tap AI can make mistakes. Check important recipe details.
      </p>
    </div>
  );
}
