/* Static, chapter-specific source reader. Ordinary links remain a no-JS fallback. */
(() => {
  'use strict';
  const marker = document.querySelector('.source-catalog');
  if (!marker || location.pathname.endsWith('/print.html')) return;
  const panel = document.createElement('aside');
  panel.id = 'source-panel';
  panel.hidden = true;
  panel.setAttribute('aria-label', 'Chapter source code');
  panel.innerHTML = `<header><strong>Source code</strong><button type="button" id="source-close" aria-label="Close source panel">Close</button></header>
    <label for="source-file">File</label><select id="source-file"></select>
    <p id="source-version"></p><nav><button type="button" id="source-copy">Copy</button> <a id="source-github" target="_blank" rel="noopener">Open on GitHub</a></nav>
    <p id="source-status" role="status"></p><pre tabindex="0" aria-label="Source file"><code id="source-code"></code></pre>`;
  document.body.append(panel);
  const select = panel.querySelector('select');
  const code = panel.querySelector('code');
  const status = panel.querySelector('#source-status');
  const close = panel.querySelector('#source-close');
  let catalogPromise;
  let contents = '';
  let returnFocus;
  let request = 0;
  async function catalog() {
    if (!catalogPromise) catalogPromise = fetch(marker.dataset.catalog).then(response => {
      if (!response.ok) throw new Error('Could not load chapter file list.');
      return response.json();
    }).catch(error => { catalogPromise = undefined; throw error; });
    return catalogPromise;
  }
  async function open(path, trigger) {
    const current = ++request;
    if (trigger) returnFocus = trigger;
    panel.hidden = false;
    document.body.classList.add('source-open');
    status.textContent = 'Loading…';
    contents = '';
    code.replaceChildren();
    try {
      const data = await catalog();
      if (current !== request) return;
      if (!data.files.includes(path)) throw new Error('File is absent from this chapter.');
      if (!select.options.length) {
        data.files.forEach(file => select.add(new Option(file, file)));
      }
      select.value = path;
      panel.querySelector('#source-version').textContent = `${data.chapter} · ${data.commit.slice(0, 12)}`;
      panel.querySelector('#source-github').href = `${data.repository}/blob/${data.commit}/${path.split('/').map(encodeURIComponent).join('/')}`;
      const url = new URL(`../source/${data.slug}/${path.split('/').map(encodeURIComponent).join('/')}.txt`, new URL(marker.dataset.catalog, location.href));
      const response = await fetch(url);
      if (!response.ok) throw new Error('Could not load this source file.');
      const text = await response.text();
      if (current !== request) return;
      contents = text;
      const extension = path.split('.').pop();
      const language = {rs: 'rust', py: 'python', toml: 'toml', json: 'json', sh: 'bash', js: 'javascript'}[extension];
      const highlighter = window.hljs;
      let highlighted;
      if (highlighter && language && highlighter.getLanguage(language)) {
        // The only HTML inserted here is escaped output from mdBook's highlighter.
        highlighted = highlighter.highlight(language, text, true).value.split('\n');
      }
      const lines = text.split('\n');
      if (lines.at(-1) === '') lines.pop();
      lines.forEach((line, index) => {
        const row = document.createElement('span');
        row.className = 'source-line';
        const number = document.createElement('span');
        number.className = 'source-line-number';
        number.setAttribute('aria-hidden', 'true');
        number.textContent = String(index + 1);
        const value = document.createElement('span');
        if (highlighted) value.innerHTML = highlighted[index] || ' ';
        else value.textContent = line || ' ';
        row.append(number, value);
        code.append(row);
      });
      status.textContent = `${lines.length} lines`;
      if (trigger) close.focus();
    } catch (error) {
      if (current === request) status.textContent = error.message;
    }
  }
  function dismiss() {
    ++request;
    panel.hidden = true;
    document.body.classList.remove('source-open');
    if (returnFocus) returnFocus.focus();
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('a.source-link');
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    open(link.dataset.source, link);
  });
  close.addEventListener('click', dismiss);
  select.addEventListener('change', () => open(select.value));
  document.addEventListener('keydown', event => {
    if (panel.hidden) return;
    if (event.key === 'Escape') { event.preventDefault(); dismiss(); }
    // The mobile drawer is modal; keep keyboard focus inside until dismissed.
    if (event.key === 'Tab' && matchMedia('(max-width: 900px)').matches) {
      const items = Array.from(panel.querySelectorAll('button, select, a[href], pre[tabindex]'));
      const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  panel.querySelector('#source-copy').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(contents); status.textContent = 'Copied source code.'; }
    catch (_) { status.textContent = 'Copy unavailable. Select the source code to copy it.'; }
  });
})();
