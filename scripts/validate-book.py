#!/usr/bin/env python3
"""Check imported chapter order, snapshot identity, and local source targets."""
import json
import hashlib
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / 'chapters.json').read_text())
summary = (root / 'docs/SUMMARY.md').read_text()
expected = [f"chapters/{c['slug']}.md" for c in manifest['chapters']]
actual = re.findall(r'^- \[[^\]]+\]\(([^)]+)\)', summary, re.M)
assert actual == expected, 'Summary chapter order differs from manifest'
count = 0
snapshots = []
for chapter in manifest['chapters']:
    page = root / 'docs/chapters' / (chapter['slug'] + '.md')
    text = page.read_text()
    catalog = json.loads((root / 'docs/assets/catalog' / (chapter['slug'] + '.json')).read_text())
    assert catalog['commit'] == chapter['commit']
    for file in catalog['files']:
        snapshot = root / 'docs/assets/source' / chapter['slug'] / (file + '.txt')
        if 'source_dir' in chapter:
            original = root / chapter['source_dir'] / file
            assert snapshot.read_bytes() == original.read_bytes(), f'Snapshot mismatch: {file}'
            assert hashlib.sha256(original.read_bytes()).hexdigest() == chapter['source_hashes'][file], f'Checksum mismatch: {file}'
        else:
            snapshots.append((chapter['commit'] + ':' + file, snapshot))
    if 'comparisons' in chapter:
        comparisons = json.loads((root / chapter['comparisons']).read_text())
        assert text.count('<div class="code-comparison">') == len(comparisons)
        assert '{{compare:' not in text, f'Unexpanded comparison in {page}'
        covered = {}
        for snippets in comparisons.values():
            assert [s['label'] for s in snippets] == ['Rust', 'Python', 'JavaScript']
            for snippet in snippets:
                covered.setdefault(snippet['path'], set()).update(range(snippet['start'], snippet['end'] + 1))
        # The walkthrough must account for every code line after the header.
        for path, lines in covered.items():
            source = (root / chapter['source_dir'] / path).read_text().splitlines()
            for number, line in enumerate(source, 1):
                if number >= min(lines) and line.strip() and not line.lstrip().startswith(('//', '# ')):
                    assert number in lines, f'Code omitted from walkthrough: {path}:{number}'
        assert catalog['repository'] == chapter['repository']
    for href in re.findall(r'class="source-link" href="([^"]+)"', text):
        assert (page.parent / unquote(href)).is_file(), f'Missing source target: {href}'
        count += 1
    assert catalog['references'] or chapter['number'] == 15, f'No source links: {page}'
intro = (root / 'docs/introduction.md').read_text()
intro_links = re.findall(r'href="(chapters/[^"#]+)"|\]\((chapters/[^)#]+)\)', intro)
intro_links = [html_link or markdown_link for html_link, markdown_link in intro_links]
assert intro_links, 'Introduction has no chapter links'
for link in intro_links:
    assert (root / 'docs' / Path(link).with_suffix('.md')).is_file(), f'Missing intro target: {link}'
index_links = re.findall(r'class="chapter-link" href="([^"]+)"', intro)
assert index_links == [str(Path(link).with_suffix('.html')) for link in expected], 'Contents chapter order differs from manifest'
recommended_links = re.findall(r'\]\((chapters/[^)#]+)\)', intro.split('<section class="reading-path"', 1)[1].split('</section>', 1)[0])
assert recommended_links == [f"chapters/{c['slug']}.html" for c in manifest['chapters'] if c['recommended']], 'Recommended path differs from manifest'
print(f"Validated {len(expected)} chapters, exact source snapshots, and {count} source links")

# Read all pinned Git blobs in one process rather than spawning per file.
blobs = subprocess.check_output(['git', 'cat-file', '--batch'],
    input=('\n'.join(ref for ref, _ in snapshots) + '\n').encode(), cwd=root)
position = 0
for ref, snapshot in snapshots:
    end = blobs.index(b'\n', position)
    header = blobs[position:end].split()
    assert len(header) == 3 and header[1] == b'blob', f'Missing Git blob: {ref}'
    size = int(header[2])
    position = end + 1
    assert snapshot.read_bytes() == blobs[position:position + size], f'Snapshot mismatch: {snapshot}'
    position += size + 1
print(f'Verified {len(snapshots)} source files against their pinned Git blobs')
