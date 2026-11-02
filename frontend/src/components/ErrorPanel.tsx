export function ErrorPanel({
  error,
}: {
  error: string;
}) {
  return (
    <div className="error-panel">
      <strong>Backend connection required</strong>
      <p>{error}</p>
      <code>python -m uvicorn backend.web:app --port 8000</code>
    </div>
  );
}
