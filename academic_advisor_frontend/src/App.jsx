// src/App.jsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import StartPage from "./pages/StartPage";
import SetupPage from "./pages/SetupPage";
import ChatPage from "./pages/ChatPage";
import ResultsPage from "./pages/ResultsPage.jsx";


export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<StartPage />} />
          <Route path="/setup" element={<SetupPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/chat/:id" element={<ChatPage />} />
          <Route path="/results/:id" element={<ResultsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
