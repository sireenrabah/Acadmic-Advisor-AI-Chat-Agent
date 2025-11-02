// src/hooks/useLocal.js
import { useEffect, useState } from "react";

export default function useLocal(key, initial) {
  const [v, setV] = useState(() => {
    const s = localStorage.getItem(key);
    if (s === null) return initial;
    // Try JSON first; if it fails, fall back to the raw string.
    try { return JSON.parse(s); } catch { return s; }
  });

  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(v)); } catch {}
  }, [key, v]);

  return [v, setV];
}
