/**
 * Renders the LLM-generated incident report. The model emits a small Markdown subset
 * (**bold**, "## heading", and 1./- lists); a <pre> would show that markup literally, so
 * this parses just that subset into real elements rather than pulling in a full Markdown
 * dependency. Anything it doesn't recognise falls through as a plain paragraph.
 */

// Split a line into text/bold runs on **...** and render <strong> for the bold ones.
function renderInline(text, keyBase) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${keyBase}-${i}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${keyBase}-${i}`}>{part}</span>;
  });
}

function isHeadingLine(line) {
  const t = line.trim();
  // "## Heading" or a line that is entirely bold, e.g. "**Recommendations:**"
  return t.startsWith("#") || (/^\*\*[^*]+\*\*:?$/.test(t));
}

function headingText(line) {
  return line.trim().replace(/^#+\s*/, "").replace(/^\*\*|\*\*:?$/g, "").replace(/:$/, "");
}

function isListItem(line) {
  return /^\s*(\d+\.|[-*])\s+/.test(line);
}

function listItemText(line) {
  return line.replace(/^\s*(\d+\.|[-*])\s+/, "");
}

export default function IncidentReport({ text }) {
  const lines = text.split("\n");
  const blocks = [];
  let para = [];
  let list = [];

  const flushPara = () => {
    if (para.length) {
      const key = `p-${blocks.length}`;
      blocks.push(<p key={key} className="report-para">{renderInline(para.join(" "), key)}</p>);
      para = [];
    }
  };
  const flushList = () => {
    if (list.length) {
      const key = `l-${blocks.length}`;
      blocks.push(
        <ul key={key} className="report-list">
          {list.map((item, i) => <li key={`${key}-${i}`}>{renderInline(item, `${key}-${i}`)}</li>)}
        </ul>
      );
      list = [];
    }
  };

  for (const line of lines) {
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if (isHeadingLine(line)) {
      flushPara(); flushList();
      const key = `h-${blocks.length}`;
      blocks.push(<h4 key={key} className="report-heading">{headingText(line)}</h4>);
    } else if (isListItem(line)) {
      flushPara();
      list.push(listItemText(line));
    } else {
      flushList();
      para.push(line.trim());
    }
  }
  flushPara();
  flushList();

  return <div className="incident-report">{blocks}</div>;
}
