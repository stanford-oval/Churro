(function () {
  const SCORE_FORMATTER = new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

  const COLUMN_DEFINITIONS = [
    { key: "modelName", label: "Model", numeric: false },
    { key: "printed", label: "Printed", numeric: true },
    { key: "handwritten", label: "Handwritten", numeric: true },
    { key: "total", label: "Total", numeric: true },
  ];

  function contentRoot() {
    return document.documentElement.dataset.content_root || "";
  }

  function resolvePath(path) {
    if (!path) {
      return "";
    }
    if (/^(?:[a-z]+:)?\/\//i.test(path)) {
      return path;
    }
    return `${contentRoot()}${path}`;
  }

  function formatScore(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "—";
    }
    return SCORE_FORMATTER.format(value);
  }

  function compareRows(left, right, sortState) {
    const { key, direction } = sortState;
    const multiplier = direction === "asc" ? 1 : -1;
    const leftValue = left[key];
    const rightValue = right[key];

    if (typeof leftValue === "number" && typeof rightValue === "number") {
      if (leftValue !== rightValue) {
        return (leftValue - rightValue) * multiplier;
      }
      return left.modelName.localeCompare(right.modelName);
    }

    return String(leftValue || "").localeCompare(String(rightValue || "")) * multiplier;
  }

  function createModelCell(row, logoPath) {
    const wrapper = document.createElement("div");
    wrapper.className = "benchmark-model";

    if (row.hasIcon) {
      const icon = document.createElement("img");
      icon.className = "benchmark-model-icon";
      icon.alt = `${row.modelName} icon`;
      icon.src = logoPath;
      wrapper.append(icon);
    }

    const textBlock = document.createElement("div");
    textBlock.className = "benchmark-model-text";

    const nameNode = row.modelUrl ? document.createElement("a") : document.createElement("span");
    nameNode.className = "benchmark-model-name";
    nameNode.textContent = row.modelName;
    if (row.modelUrl) {
      nameNode.href = row.modelUrl;
      nameNode.target = "_blank";
      nameNode.rel = "noreferrer noopener";
    }

    textBlock.append(nameNode);

    if (row.modelId) {
      const metaNode = document.createElement("code");
      metaNode.className = "benchmark-model-id";
      metaNode.textContent = row.modelId;
      textBlock.append(metaNode);
    }

    wrapper.append(textBlock);
    return wrapper;
  }

  function createHeaderButton(column, sortState, onSort) {
    const button = document.createElement("button");
    button.className = "benchmark-sort-button";
    button.type = "button";
    button.textContent = column.label;
    button.dataset.sortKey = column.key;

    const indicator = document.createElement("span");
    indicator.className = "benchmark-sort-indicator";
    if (sortState.key === column.key) {
      indicator.textContent = sortState.direction === "asc" ? "▲" : "▼";
    } else {
      indicator.textContent = "↕";
    }
    button.append(indicator);
    button.addEventListener("click", () => onSort(column.key));
    return button;
  }

  function renderLeaderboard(container, rows, sortState, logoPath) {
    container.replaceChildren();

    const sortedRows = [...rows].sort((left, right) => compareRows(left, right, sortState));

    const wrapper = document.createElement("div");
    wrapper.className = "benchmark-table-wrapper";

    const table = document.createElement("table");
    table.className = "benchmark-table";

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    const rankHeader = document.createElement("th");
    rankHeader.scope = "col";
    rankHeader.textContent = "#";
    headerRow.append(rankHeader);

    const handleSort = (key) => {
      if (sortState.key === key) {
        sortState.direction = sortState.direction === "desc" ? "asc" : "desc";
      } else {
        sortState.key = key;
        sortState.direction = key === "modelName" ? "asc" : "desc";
      }
      renderLeaderboard(container, rows, sortState, logoPath);
    };

    for (const column of COLUMN_DEFINITIONS) {
      const th = document.createElement("th");
      th.scope = "col";
      th.append(createHeaderButton(column, sortState, handleSort));
      headerRow.append(th);
    }

    thead.append(headerRow);
    table.append(thead);

    const tbody = document.createElement("tbody");

    sortedRows.forEach((row, index) => {
      const tr = document.createElement("tr");
      if (row.hasIcon) {
        tr.classList.add("is-featured");
      }

      const rankCell = document.createElement("td");
      rankCell.className = "benchmark-rank";
      rankCell.textContent = String(index + 1);
      tr.append(rankCell);

      const modelCell = document.createElement("td");
      modelCell.append(createModelCell(row, logoPath));
      tr.append(modelCell);

      for (const key of ["printed", "handwritten", "total"]) {
        const scoreCell = document.createElement("td");
        scoreCell.className = "benchmark-score";
        scoreCell.textContent = formatScore(row[key]);
        tr.append(scoreCell);
      }

      tbody.append(tr);
    });

    table.append(tbody);
    wrapper.append(table);
    container.append(wrapper);
  }

  function renderError(container, message) {
    container.replaceChildren();
    const error = document.createElement("p");
    error.className = "benchmark-error";
    error.textContent = message;
    container.append(error);
  }

  async function initializeLeaderboard(container) {
    const logoPath = resolvePath("_static/img/churro.png");

    try {
      const response = await fetch(resolvePath("_static/data/benchmark_results.json"));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const rows = await response.json();
      const sortState = { key: "total", direction: "desc" };
      renderLeaderboard(container, rows, sortState, logoPath);
    } catch (error) {
      renderError(container, "Unable to load the benchmark leaderboard data.");
      console.error("[benchmark-leaderboard] failed to initialize", error);
    }
  }

  function initializeAll() {
    document.querySelectorAll(".benchmark-leaderboard").forEach((container) => {
      initializeLeaderboard(container);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeAll);
  } else {
    initializeAll();
  }
})();
