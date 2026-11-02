/** Runs of consecutive true values as [start, end) in ms; a run of one bin is one bin wide. */
export function runs(t: number[], on: Array<boolean | null>, step = 300_000): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  let from = -1;
  on.forEach((v, i) => {
    if (v && from < 0) from = i;
    if (!v && from >= 0) {
      out.push([t[from], t[i - 1] + step]);
      from = -1;
    }
  });
  if (from >= 0) out.push([t[from], t[on.length - 1] + step]);
  return out;
}
