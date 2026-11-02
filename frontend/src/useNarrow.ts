import { useEffect, useState } from "react";

/** True on phones and tablets (up to 820 px), where charts get less room. */
export function useNarrow() {
  const query = "(max-width: 820px)";
  const [narrow, setNarrow] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const q = window.matchMedia(query);
    const on = () => setNarrow(q.matches);
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, []);
  return narrow;
}
