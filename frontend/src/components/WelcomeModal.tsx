import { useState, useEffect } from "react";
import { X } from "lucide-react";

export function WelcomeModal() {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem("hasSeenWelcome");
    if (!hasSeenWelcome) {
      setIsOpen(true);
    }
  }, []);

  const closeModal = () => {
    setIsOpen(false);
    localStorage.setItem("hasSeenWelcome", "true");
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-300">
      <div className="bg-stone-900 border border-stone-800 rounded-xl shadow-2xl max-w-md w-full overflow-hidden animate-in zoom-in-95 duration-300">
        <div className="relative p-6 md:p-8">
          <button
            onClick={closeModal}
            className="absolute top-4 right-4 text-stone-400 hover:text-white transition-colors"
            aria-label="Close"
          >
            <X size={20} />
          </button>

          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-amber-500">
              Welcome to RAG-on-Tap! 🍻
            </h2>

            <div className="space-y-3 text-stone-300 leading-relaxed">
              <p>
                Beer RAG is your intelligent brewing companion. It uses
                Retrieval-Augmented Generation to help you discover beer
                recipes, master brewing styles, and understand ingredients with
                precision.
              </p>
              <p>
                Whether you're a novice homebrewer or a seasoned pro, our AI is
                here to tap into a deep well of brewing knowledge for you.
              </p>
              <div className="pt-2">
                <p className="text-sm text-stone-500 italic">
                  Note: To respect your privacy, no chat data is persisted in
                  our systems for more than 7 days.
                </p>
              </div>
            </div>

            <button
              onClick={closeModal}
              className="w-full mt-6 bg-amber-600 hover:bg-amber-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg shadow-amber-900/20"
            >
              Start Brewing
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
