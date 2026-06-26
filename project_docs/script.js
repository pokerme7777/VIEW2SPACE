const header = document.querySelector("[data-site-header]");
const navToggle = document.querySelector("[data-nav-toggle]");
const navLinks = document.querySelector("[data-nav-links]");
const navAnchors = Array.from(document.querySelectorAll(".nav-links a"));
const copyButton = document.querySelector("[data-copy-target]");
const copyStatus = document.querySelector("[data-copy-status]");

const setScrolledHeader = () => {
  if (!header) return;
  header.classList.toggle("is-scrolled", window.scrollY > 8);
};

setScrolledHeader();
window.addEventListener("scroll", setScrolledHeader, { passive: true });

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const isOpen = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!isOpen));
    document.body.classList.toggle("nav-open", !isOpen);
  });

  navAnchors.forEach((anchor) => {
    anchor.addEventListener("click", () => {
      navToggle.setAttribute("aria-expanded", "false");
      document.body.classList.remove("nav-open");
    });
  });
}

const observedSections = navAnchors
  .map((anchor) => {
    const id = anchor.getAttribute("href");
    return id ? document.querySelector(id) : null;
  })
  .filter(Boolean);

if ("IntersectionObserver" in window && observedSections.length > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) return;

      navAnchors.forEach((anchor) => {
        anchor.classList.toggle("is-active", anchor.getAttribute("href") === `#${visible.target.id}`);
      });
    },
    { rootMargin: "-18% 0px -64% 0px", threshold: [0.1, 0.35, 0.6] },
  );

  observedSections.forEach((section) => observer.observe(section));
}

if (copyButton && copyStatus) {
  copyButton.addEventListener("click", async () => {
    const targetId = copyButton.getAttribute("data-copy-target");
    const target = targetId ? document.getElementById(targetId) : null;
    const text = target ? target.innerText.trim() : "";

    try {
      await navigator.clipboard.writeText(text);
      copyStatus.textContent = "BibTeX copied.";
      copyStatus.classList.remove("is-error");
    } catch {
      copyStatus.textContent = "Copy failed. Select the citation text manually.";
      copyStatus.classList.add("is-error");
    }
  });
}
