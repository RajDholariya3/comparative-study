import React, { useState } from "react";

const App = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:3000/analyze");
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error("Error fetching analysis:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-start min-h-screen bg-gradient-to-br from-blue-900 to-purple-900 p-10 text-white">
      <div className="bg-white text-gray-900 p-8 rounded-2xl shadow-2xl w-full max-w-3xl text-center">
        <h2 className="text-3xl font-extrabold text-blue-600 mb-6">🚀 ML Model Analysis</h2>

        <button
          onClick={handleAnalyze}
          className="bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-full text-lg font-semibold transition-all shadow-lg"
        >
          🔍 Start Analysis
        </button>

        {loading && <p className="mt-4 text-blue-500 font-medium">⏳ Running analysis... Please wait!</p>}

        {data && (
          <div className="mt-8 space-y-6">
            <div className="bg-gray-100 p-6 rounded-xl shadow-md">
              <h3 className="text-xl font-semibold text-gray-700">📂 Dataset Overview</h3>
              <div className="mt-3 flex justify-between text-lg font-medium text-gray-800">
                <span>📊 Rows: {data.dataset_info.shape[0]}</span>
                <span>📈 Columns: {data.dataset_info.shape[1]}</span>
              </div>
            </div>

            <div className="p-5 border-2 border-purple-500 text-purple-700 rounded-xl bg-purple-100 shadow-md">
              ⭐ Best Model: <strong>{data.best_model.model}</strong> with Accuracy:{" "}
              <strong>{data.best_model.accuracy}%</strong>
            </div>

            <h3 className="text-2xl font-bold text-gray-800">📊 Model Performance</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {data.results.map((result, index) => (
                <div key={index} className="bg-gray-200 p-4 rounded-lg shadow-lg">
                  <h4 className="text-lg font-bold text-blue-600">{result.model}</h4>
                  <p className="text-gray-700 mt-2">
                    <strong>🎯 Accuracy:</strong> {result.accuracy}%
                  </p>
                  <p className="text-gray-700">
                    <strong>📌 Precision:</strong> {result.precision}%
                  </p>
                  <p className="text-gray-700">
                    <strong>🔄 Recall:</strong> {result.recall}%
                  </p>
                  <p className="text-gray-700">
                    <strong>⚖️ F1 Score:</strong> {result.f1_score}%
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
