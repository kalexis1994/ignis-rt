/**
 * Fix Mermaid diagram contrast using Ignis palette.
 * Injects styles into SVG, then fixes text color per-node based on
 * actual background fill (WCAG luminance calculation).
 */
(function() {
  function lin(c) { return c <= 0.04045 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4); }
  function lum(r,g,b) { return 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b); }
  function parseCol(s) {
    if (!s) return null;
    var m = s.match(/#([0-9a-f]{3,8})/i);
    if (m) { var h=m[1]; if(h.length===3)h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
      return [parseInt(h.slice(0,2),16)/255,parseInt(h.slice(2,4),16)/255,parseInt(h.slice(4,6),16)/255]; }
    m = s.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
    if (m) return [m[1]/255,m[2]/255,m[3]/255];
    return null;
  }

  function isDark() {
    return (document.body||document.documentElement).getAttribute('data-md-color-scheme') === 'slate';
  }

  function fix() {
    var svgs = document.querySelectorAll('svg[id^="__mermaid"]');
    if (!svgs.length) return false;

    svgs.forEach(function(svg) {
      // Inject palette into SVG stylesheet (for edges, markers, clusters)
      if (!svg.dataset.ignisCss) {
        svg.dataset.ignisCss = '1';
        var id = svg.id;
        var dark = isDark();
        var edgeColor = dark ? '#8FA7C4' : '#4A6380';
        var clusterFill = dark ? '#1C1C1E' : '#f0f0f0';
        var clusterStroke = dark ? '#636366' : '#8FA7C4';
        var clusterText = dark ? '#DCE4ED' : '#1C1C1E';

        var css = '\n/* Ignis */\n' +
          '#'+id+' .edgePath .path, #'+id+' .flowchart-link { stroke:'+edgeColor+' !important; }\n' +
          '#'+id+' marker path { fill:'+edgeColor+' !important; }\n' +
          '#'+id+' .cluster rect { fill:'+clusterFill+' !important; stroke:'+clusterStroke+' !important; }\n' +
          '#'+id+' .cluster text, #'+id+' .cluster span, #'+id+' .cluster-label span { fill:'+clusterText+' !important; color:'+clusterText+' !important; }\n';

        var styleEl = svg.querySelector('style');
        if (styleEl) styleEl.textContent += css;
      }

      // Per-node contrast: read rect fill, compute luminance, set text color
      if (!svg.dataset.ignisContrast) {
        svg.dataset.ignisContrast = '1';
        svg.querySelectorAll('g.node').forEach(function(node) {
          var rect = node.querySelector('rect');
          if (!rect) return;
          // Get fill from inline style or attribute or computed
          var fillStr = rect.getAttribute('style') || '';
          var c = parseCol(fillStr);
          if (!c) c = parseCol(rect.getAttribute('fill'));
          if (!c) c = parseCol(getComputedStyle(rect).fill);
          if (!c) return;
          var textCol = lum(c[0],c[1],c[2]) > 0.179 ? '#000' : '#fff';
          // Set on all text descendants
          node.querySelectorAll('.nodeLabel, .nodeLabel p, span').forEach(function(el) {
            el.style.setProperty('color', textCol, 'important');
          });
          node.querySelectorAll('.label').forEach(function(el) {
            el.style.setProperty('color', textCol, 'important');
          });
        });
      }
    });
    return true;
  }

  var n = 0;
  function poll() { if (fix() || ++n > 20) return; setTimeout(poll, 500); }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() { setTimeout(poll, 800); });
  } else { setTimeout(poll, 800); }
  if (typeof document$ !== 'undefined') {
    document$.subscribe(function() { n=0; setTimeout(poll, 800); });
  }
})();
