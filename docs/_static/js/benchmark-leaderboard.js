(function () {
  const SCORE_FORMATTER = new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  const GROUP_DEFINITIONS = [
    { key: "printed", label: "Printed", metricType: "print" },
    { key: "handwritten", label: "Handwritten", metricType: "handwriting" },
  ];
  const DEFAULT_SORT_STATE = { key: "total", direction: "desc" };
  const DEFAULT_EXPANDED_GROUPS = { printed: false, handwritten: false };

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

  function getGroupDefinition(groupKey) {
    return GROUP_DEFINITIONS.find((group) => group.key === groupKey) || null;
  }

  function getGroupForColumn(columnKey) {
    const groupKey = columnKey.includes(":") ? columnKey.split(":")[0] : columnKey;
    return getGroupDefinition(groupKey);
  }

  function getLanguageMetrics(row, group) {
    return row.main_language_and_type_metrics?.[group.metricType] || {};
  }

  function getGroupLanguages(rows) {
    const languagesByGroup = {};

    GROUP_DEFINITIONS.forEach((group) => {
      const languages = new Set();
      rows.forEach((row) => {
        Object.keys(getLanguageMetrics(row, group)).forEach((language) => {
          languages.add(language);
        });
      });
      languagesByGroup[group.key] = [...languages].sort((left, right) => left.localeCompare(right));
    });

    return languagesByGroup;
  }

  function getColumnValue(row, columnKey) {
    if (columnKey === "modelName") {
      return row.modelName;
    }
    if (columnKey === "printed" || columnKey === "handwritten" || columnKey === "total") {
      return row[columnKey];
    }

    const [groupKey, language] = columnKey.split(":");
    const group = getGroupDefinition(groupKey);
    if (!group || !language) {
      return undefined;
    }

    return getLanguageMetrics(row, group)[language];
  }

  function isNumericColumn(columnKey) {
    return columnKey !== "modelName";
  }

  function createIconFrame(row) {
    const iconFrame = document.createElement("span");
    iconFrame.className = "benchmark-model-icon-frame";

    if (!row.iconPath) {
      const placeholder = document.createElement("span");
      placeholder.className = "benchmark-model-icon-placeholder";
      placeholder.setAttribute("aria-hidden", "true");
      iconFrame.append(placeholder);
      return iconFrame;
    }

    const icon = document.createElement("img");
    icon.className = "benchmark-model-icon";
    icon.alt = `${row.modelName} icon`;
    icon.src = resolvePath(row.iconPath);
    iconFrame.append(icon);

    return iconFrame;
  }

  function compareRows(left, right, sortState) {
    const { key, direction } = sortState;
    const multiplier = direction === "asc" ? 1 : -1;
    const leftValue = getColumnValue(left, key);
    const rightValue = getColumnValue(right, key);

    if (isNumericColumn(key)) {
      const leftNumber = Number(leftValue);
      const rightNumber = Number(rightValue);

      if (Number.isNaN(leftNumber) && !Number.isNaN(rightNumber)) {
        return 1;
      }
      if (!Number.isNaN(leftNumber) && Number.isNaN(rightNumber)) {
        return -1;
      }
      if (!Number.isNaN(leftNumber) && !Number.isNaN(rightNumber) && leftNumber !== rightNumber) {
        return (leftNumber - rightNumber) * multiplier;
      }
      return left.modelName.localeCompare(right.modelName);
    }

    return String(leftValue || "").localeCompare(String(rightValue || "")) * multiplier;
  }

  function createModelCell(row) {
    const wrapper = document.createElement("div");
    wrapper.className = "benchmark-model";

    wrapper.append(createIconFrame(row));

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

  function createHeaderButton(columnKey, label, sortState, onSort) {
    const button = document.createElement("button");
    button.className = "benchmark-sort-button";
    button.type = "button";
    button.textContent = label;
    button.dataset.sortKey = columnKey;

    const indicator = document.createElement("span");
    indicator.className = "benchmark-sort-indicator";
    if (sortState.key === columnKey) {
      indicator.textContent = sortState.direction === "asc" ? "▲" : "▼";
    } else {
      indicator.textContent = "↕";
    }
    button.append(indicator);
    button.addEventListener("click", () => onSort(columnKey));
    return button;
  }

  function createGroupToggle(group, expanded, languageCount, onToggle) {
    const button = document.createElement("button");
    button.className = "benchmark-group-toggle";
    button.type = "button";
    button.textContent = expanded ? "Collapse" : "Expand";
    button.disabled = languageCount === 0;
    button.setAttribute("aria-expanded", expanded ? "true" : "false");
    button.setAttribute(
      "aria-label",
      expanded
        ? `Collapse ${group.label.toLowerCase()} language columns`
        : `Expand ${group.label.toLowerCase()} language columns (${languageCount} languages)`,
    );
    button.title =
      languageCount === 0
        ? `No per-language ${group.label.toLowerCase()} scores available`
        : expanded
          ? `Hide ${group.label.toLowerCase()} language scores`
          : `Show ${languageCount} ${group.label.toLowerCase()} language scores`;
    button.addEventListener("click", () => onToggle(group.key));
    return button;
  }

  function createStandaloneHeaderCell(columnKey, label, sortState, onSort) {
    const th = document.createElement("th");
    th.scope = "col";
    th.append(createHeaderButton(columnKey, label, sortState, onSort));
    return th;
  }

  function createFlatGroupHeaderCell(group, sortState, onSort, onToggle, expanded, languageCount) {
    const th = document.createElement("th");
    th.scope = "col";

    const wrapper = document.createElement("div");
    wrapper.className = "benchmark-group-header";
    wrapper.append(createHeaderButton(group.key, group.label, sortState, onSort));
    wrapper.append(createGroupToggle(group, expanded, languageCount, onToggle));
    th.append(wrapper);

    return th;
  }

  function createExpandedGroupHeaderCell(group, onToggle, languageCount) {
    const th = document.createElement("th");
    th.scope = "colgroup";
    th.colSpan = languageCount + 1;

    const wrapper = document.createElement("div");
    wrapper.className = "benchmark-group-header benchmark-group-header--expanded";

    const label = document.createElement("span");
    label.className = "benchmark-group-label";
    label.textContent = group.label;
    wrapper.append(label);
    wrapper.append(createGroupToggle(group, true, languageCount, onToggle));

    th.append(wrapper);
    return th;
  }

  function createVisibleScoreColumns(groupLanguages, expandedGroups) {
    const visibleColumns = [];

    GROUP_DEFINITIONS.forEach((group) => {
      visibleColumns.push(group.key);
      if (expandedGroups[group.key]) {
        groupLanguages[group.key].forEach((language) => {
          visibleColumns.push(`${group.key}:${language}`);
        });
      }
    });

    visibleColumns.push("total");
    return visibleColumns;
  }

  function normalizeSortState(sortState, expandedGroups) {
    const group = getGroupForColumn(sortState.key);
    if (group && sortState.key.includes(":") && !expandedGroups[group.key]) {
      sortState.key = group.key;
    }
  }

  function renderLeaderboard(container, rows, sortState, expandedGroups) {
    normalizeSortState(sortState, expandedGroups);
    container.replaceChildren();

    const groupLanguages = getGroupLanguages(rows);
    const sortedRows = [...rows].sort((left, right) => compareRows(left, right, sortState));
    const hasExpandedGroups = GROUP_DEFINITIONS.some((group) => expandedGroups[group.key]);
    const visibleScoreColumns = createVisibleScoreColumns(groupLanguages, expandedGroups);

    const wrapper = document.createElement("div");
    wrapper.className = "benchmark-table-wrapper";

    const table = document.createElement("table");
    table.className = "benchmark-table";

    const handleSort = (key) => {
      if (sortState.key === key) {
        sortState.direction = sortState.direction === "desc" ? "asc" : "desc";
      } else {
        sortState.key = key;
        sortState.direction = key === "modelName" ? "asc" : "desc";
      }
      renderLeaderboard(container, rows, sortState, expandedGroups);
    };

    const handleToggle = (groupKey) => {
      expandedGroups[groupKey] = !expandedGroups[groupKey];
      normalizeSortState(sortState, expandedGroups);
      renderLeaderboard(container, rows, sortState, expandedGroups);
    };

    const thead = document.createElement("thead");
    const topHeaderRow = document.createElement("tr");

    const rankHeader = document.createElement("th");
    rankHeader.scope = "col";
    rankHeader.textContent = "#";
    if (hasExpandedGroups) {
      rankHeader.rowSpan = 2;
    }
    topHeaderRow.append(rankHeader);

    const modelHeader = createStandaloneHeaderCell("modelName", "Model", sortState, handleSort);
    if (hasExpandedGroups) {
      modelHeader.rowSpan = 2;
    }
    topHeaderRow.append(modelHeader);

    if (!hasExpandedGroups) {
      GROUP_DEFINITIONS.forEach((group) => {
        topHeaderRow.append(
          createFlatGroupHeaderCell(
            group,
            sortState,
            handleSort,
            handleToggle,
            false,
            groupLanguages[group.key].length,
          ),
        );
      });
      topHeaderRow.append(createStandaloneHeaderCell("total", "Total", sortState, handleSort));
      thead.append(topHeaderRow);
    } else {
      GROUP_DEFINITIONS.forEach((group) => {
        if (expandedGroups[group.key]) {
          topHeaderRow.append(
            createExpandedGroupHeaderCell(group, handleToggle, groupLanguages[group.key].length),
          );
          return;
        }

        const flatHeader = createFlatGroupHeaderCell(
          group,
          sortState,
          handleSort,
          handleToggle,
          false,
          groupLanguages[group.key].length,
        );
        flatHeader.rowSpan = 2;
        topHeaderRow.append(flatHeader);
      });

      const totalHeader = createStandaloneHeaderCell("total", "Total", sortState, handleSort);
      totalHeader.rowSpan = 2;
      topHeaderRow.append(totalHeader);
      thead.append(topHeaderRow);

      const subheaderRow = document.createElement("tr");
      GROUP_DEFINITIONS.forEach((group) => {
        if (!expandedGroups[group.key]) {
          return;
        }

        const overallHeader = document.createElement("th");
        overallHeader.className = "benchmark-subcolumn";
        overallHeader.scope = "col";
        overallHeader.append(createHeaderButton(group.key, "Overall", sortState, handleSort));
        subheaderRow.append(overallHeader);

        groupLanguages[group.key].forEach((language) => {
          const languageHeader = document.createElement("th");
          languageHeader.className = "benchmark-subcolumn";
          languageHeader.scope = "col";
          languageHeader.append(
            createHeaderButton(`${group.key}:${language}`, language, sortState, handleSort),
          );
          subheaderRow.append(languageHeader);
        });
      });
      thead.append(subheaderRow);
    }
    table.append(thead);

    const tbody = document.createElement("tbody");

    sortedRows.forEach((row, index) => {
      const tr = document.createElement("tr");
      if (row.iconPath) {
        tr.classList.add("is-featured");
      }

      const rankCell = document.createElement("td");
      rankCell.className = "benchmark-rank";
      rankCell.textContent = String(index + 1);
      tr.append(rankCell);

      const modelCell = document.createElement("td");
      modelCell.append(createModelCell(row));
      tr.append(modelCell);

      for (const key of visibleScoreColumns) {
        const scoreCell = document.createElement("td");
        scoreCell.className = "benchmark-score";
        scoreCell.textContent = formatScore(getColumnValue(row, key));
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
    try {
      const response = await fetch(resolvePath("_static/data/benchmark_results.json"));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const rows = await response.json();
      const sortState = { ...DEFAULT_SORT_STATE };
      const expandedGroups = { ...DEFAULT_EXPANDED_GROUPS };
      renderLeaderboard(container, rows, sortState, expandedGroups);
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
