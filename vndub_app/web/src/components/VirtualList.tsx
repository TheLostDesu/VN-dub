import { useEffect, useRef, useState, type ReactNode } from "react";

/** A small fixed-row-height virtual list: only the visible rows are in the DOM. */
export function VirtualList<T>({
  items,
  rowHeight,
  renderRow,
  keyOf,
  scrollToIndex,
  label,
  testid,
  className,
}: {
  items: T[];
  rowHeight: number;
  renderRow: (item: T, index: number) => ReactNode;
  keyOf: (item: T) => string;
  /** when this changes the row is scrolled into view */
  scrollToIndex?: number | null;
  label: string;
  testid?: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(480);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setHeight(el.clientHeight || 480);
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight || 480));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || scrollToIndex === null || scrollToIndex === undefined || scrollToIndex < 0) return;
    const top = scrollToIndex * rowHeight;
    if (top < el.scrollTop) el.scrollTop = top;
    else if (top + rowHeight > el.scrollTop + el.clientHeight) el.scrollTop = top + rowHeight - el.clientHeight;
  }, [scrollToIndex, rowHeight]);

  const overscan = 6;
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(items.length, Math.ceil((scrollTop + height) / rowHeight) + overscan);
  const slice = items.slice(start, end);

  return (
    <div ref={ref} className={`virtual-list ${className ?? ""}`} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)} role="list" aria-label={label} data-testid={testid}>
      <div style={{ height: items.length * rowHeight, position: "relative" }}>
        {slice.map((item, i) => (
          <div key={keyOf(item)} role="listitem" style={{ position: "absolute", top: (start + i) * rowHeight, height: rowHeight, left: 0, right: 0 }}>
            {renderRow(item, start + i)}
          </div>
        ))}
      </div>
    </div>
  );
}
