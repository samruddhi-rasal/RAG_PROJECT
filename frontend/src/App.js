import React, { useState } from "react";
import axios from "axios";
import "./App.css";
// Main App component
function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/ask",
        { question }
      );

      setResult(response.data);
    } catch (error) {
      setResult({ error: "Backend error" });
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1>RAG Chat System 🚀</h1>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question..."
      />

      <button onClick={askQuestion}>
        {loading ? "Thinking..." : "Ask"}
      </button>

      {result && (
        <div className="answer">
          <h3>Answer:</h3>

          {result.error ? (
            <p>{result.error}</p>
          ) : (
            <>
              <p>{result.answer}</p>

              <p style={{ marginTop: "10px" }}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;