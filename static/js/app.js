// Shared navigation behavior only.
(function () {
  const path = location.pathname || "/";
  const links = document.querySelectorAll("[data-nav]");

  links.forEach((link) => {
    if (link.getAttribute("href") === path) {
      link.classList.add("active");
    }
  });

  if (path === "/") {
    document.addEventListener("keydown", (event) => {
      const tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (event.key === "1") location.href = "/sign-to-text";
      if (event.key === "2") location.href = "/text-to-sign";
      if (event.key === "3") location.href = "/video-lookup";
    });
  }
})();
