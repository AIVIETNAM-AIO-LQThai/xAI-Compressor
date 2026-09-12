export function LoadingPanel({
  label = "Loading evidence...",
}: {
  label?: string;
}) {
  return (
    <div className="loading-panel">
      <div className="scanner-line" />
      <div className="loading-orbit" />
      <p>{label}</p>
    </div>
  );
}
