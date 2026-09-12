export function ErrorPanel({
  error,
}: {
  error: string;
}) {
  return (
    <div className="error-panel">
      <strong>Backend connection required</strong>
      <p>{error}</p>
      <code>
        python -m uvicorn backend.app.main:app --reload
      </code>
    </div>
  );
}
