/* Scroll-reveal for landing page sections and hero-page body class */
function setupReveal() {
  /* Mark the homepage so CSS can target it (hide default footer, etc.) */
  if (document.querySelector(".hero")) {
    document.documentElement.classList.add("hero-page");
    document.body.classList.add("hero-page");
  }

  /* Reset and re-observe landing sections */
  var sections = document.querySelectorAll(".landing-section");
  if (!sections.length) return;

  sections.forEach(function (section) {
    section.classList.remove("visible");
  });

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  sections.forEach(function (section) {
    observer.observe(section);
  });
}

/* Material theme's instant navigation uses the document$ RxJS observable */
if (typeof document$ !== "undefined") {
  document$.subscribe(setupReveal);
} else {
  document.addEventListener("DOMContentLoaded", setupReveal);
}
