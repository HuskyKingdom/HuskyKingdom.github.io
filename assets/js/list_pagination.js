(function () {
  function initPaginatedList(list) {
    var items = Array.prototype.slice.call(list.querySelectorAll(".paginated-item"));
    var pagination = list.querySelector(".list-pagination");
    var pageSize = parseInt(list.getAttribute("data-page-size"), 10) || 5;
    var pageCount = Math.ceil(items.length / pageSize);
    var currentPage = 1;

    if (!pagination || pageCount <= 1) {
      if (pagination) pagination.hidden = true;
      return;
    }

    function setPage(page) {
      currentPage = Math.max(1, Math.min(page, pageCount));

      items.forEach(function (item, index) {
        var visible = index >= (currentPage - 1) * pageSize && index < currentPage * pageSize;
        item.hidden = !visible;
      });

      Array.prototype.forEach.call(pagination.querySelectorAll("button[data-page]"), function (button) {
        var isCurrent = parseInt(button.getAttribute("data-page"), 10) === currentPage;
        button.classList.toggle("active", isCurrent);
        button.setAttribute("aria-current", isCurrent ? "page" : "false");
      });

      pagination.querySelector("[data-prev]").disabled = currentPage === 1;
      pagination.querySelector("[data-next]").disabled = currentPage === pageCount;
    }

    function makeButton(label, className) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = className;
      button.textContent = label;
      return button;
    }

    var prev = makeButton("Prev", "pagination-btn pagination-prev");
    prev.setAttribute("data-prev", "");
    prev.addEventListener("click", function () {
      setPage(currentPage - 1);
    });
    pagination.appendChild(prev);

    for (var page = 1; page <= pageCount; page += 1) {
      var pageButton = makeButton(String(page), "pagination-btn pagination-page");
      pageButton.setAttribute("data-page", String(page));
      pageButton.setAttribute("aria-label", "Go to page " + page);
      pageButton.addEventListener("click", function () {
        setPage(parseInt(this.getAttribute("data-page"), 10));
      });
      pagination.appendChild(pageButton);
    }

    var next = makeButton("Next", "pagination-btn pagination-next");
    next.setAttribute("data-next", "");
    next.addEventListener("click", function () {
      setPage(currentPage + 1);
    });
    pagination.appendChild(next);

    setPage(1);
  }

  document.addEventListener("DOMContentLoaded", function () {
    Array.prototype.forEach.call(document.querySelectorAll(".paginated-list"), initPaginatedList);
  });
})();
