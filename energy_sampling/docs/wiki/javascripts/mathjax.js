window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  }
};

// Material's instant navigation swaps the page body without a reload, so the
// initial typeset only covers the first page. Re-typeset on every page change,
// chained on MathJax's own startup promise so two typesets never overlap.
// On the initial load MathJax's own startup typesets the page; typesetting it
// again nests a second copy inside the first's assistive block (the clipped
// equation). So only typeset when the current document has no rendered math.
document$.subscribe(function () {
  if (!window.MathJax || !MathJax.startup || !MathJax.startup.promise) { return; }
  MathJax.startup.promise = MathJax.startup.promise.then(function () {
    if (document.querySelector("mjx-container")) { return; }
    // The page swap dropped MathJax's stylesheet; clearing the output cache
    // makes the next typeset rebuild it instead of failing to insert rules.
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
