import type { ReactNode } from "react";

interface Props {
  eyebrow: string;
  title: string;
  description: string;
  badge?: ReactNode;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  badge,
}: Props) {
  return (
    <header className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {badge ? <div>{badge}</div> : null}
    </header>
  );
}
