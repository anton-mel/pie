#!/usr/bin/env python3
"""Rebuild generated book content from the commits in chapters.json."""
import html
import hashlib
import json
import re
import subprocess
import textwrap
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
# Tutorial snapshots belong to upstream, including when the book is built in a fork.
REPO = 'https://github.com/anton-mel/pie'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def read_blobs(commit, names):
    refs = [commit + ':' + name for name in names]
    data = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
                                   input=('\n'.join(refs) + '\n').encode())
    position = 0
    for name in names:
        end = data.index(b'\n', position)
        header = data[position:end].split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError(f'Not a source blob: {commit}:{name}')
        size = int(header[2])
        position = end + 1
        yield name, data[position:position + size]
        position += size + 1


def resolve(label, files, aliases):
    if label in aliases:
        return aliases[label]
    if label in files:
        return label
    matches = [p for p in files if p.endswith('/' + label)]
    if len(matches) == 1:
        return matches[0]
    children = [p for p in files if p.startswith(label.rstrip('/') + '/')]
    if children:
        for suffix in ('/src/main.rs', '/src/lib.rs', '/__init__.py'):
            target = label.rstrip('/') + suffix
            if target in children:
                return target
        return sorted(children)[0]
    return None


def link_sources(text, chapter, files):
    """Only rewrite inline code outside fenced blocks; retain symbols as code."""
    fenced = False
    output = []
    references = []
    for line in text.splitlines():
        if line.lstrip().startswith(('```', '~~~')):
            fenced = not fenced
        if not fenced and not line.lstrip().startswith(('```', '~~~')):
            def replace(match):
                label = match.group(1)
                target = resolve(label, files, chapter['aliases'])
                if not target:
                    return match.group(0)
                references.append({'label': label, 'path': target})
                # mdBook rewrites literal .md in links even before a .txt suffix.
                url = '../assets/source/' + chapter['slug'] + '/' + quote(target).replace('.md', '%2Emd') + '.txt'
                return (f'<a class="source-link" href="{url}" '
                        f'data-source="{html.escape(target, quote=True)}">'
                        f'<code>{html.escape(label)}</code></a>')
            line = re.sub(r'`([^`\n]+)`', replace, line)
        output.append(line)
    return '\n'.join(output) + '\n', references


def render_comparisons(text, chapter, files, references):
    """Render aligned excerpts from the same snapshots used by the source panel."""
    comparisons = json.loads((ROOT / chapter['comparisons']).read_text())
    used = []

    def render(match):
        key = match.group(1)
        used.append(key)
        columns = ['<div class="code-comparison">']
        for snippet in comparisons[key]:
            path = snippet['path']
            lines = files[path].splitlines()
            start, end = snippet['start'], snippet['end']
            if not 1 <= start <= end <= len(lines):
                raise ValueError(f'Invalid excerpt: {key} {path}:{start}-{end}')
            # Remove only whole comment lines; preserve code and its indentation.
            selected = [line for line in lines[start - 1:end]
                        if not line.lstrip().startswith(('//', '# '))]
            excerpt = textwrap.dedent('\n'.join(selected)).strip('\n')
            # Blank lines inside raw HTML otherwise end CommonMark HTML blocks.
            escaped_excerpt = html.escape(excerpt).replace('\n', '&#10;')
            url = '../assets/source/' + chapter['slug'] + '/' + quote(path) + '.txt'
            references.append({'label': snippet['label'], 'path': path})
            columns.append(
                '<section class="code-language">'
                f'<h3>{html.escape(snippet["label"])}</h3>'
                f'<a class="source-link" href="{url}" data-source="{html.escape(path, quote=True)}">'
                f'Full source · lines {start}–{end}</a>'
                f'<pre><code class="language-{snippet["language"]}">{escaped_excerpt}</code></pre>'
                '</section>')
        columns.append('</div>')
        return '\n'.join(columns)

    text = re.sub(r'\{\{compare:([\w-]+)\}\}', render, text)
    if set(used) != set(comparisons) or len(used) != len(comparisons):
        raise ValueError('Every comparison must appear exactly once in the chapter')
    return text


def main():
    manifest = json.loads((ROOT / 'chapters.json').read_text())
    chapters = manifest['chapters']
    if [c['number'] for c in chapters] != list(range(1, len(chapters) + 1)):
        raise ValueError('Chapter numbers must be consecutive and ordered')
    for directory in ('chapters', 'assets/source', 'assets/catalog'):
        (DOCS / directory).mkdir(parents=True, exist_ok=True)
    parts = {1: 'Inferlets and KV memory', 6: 'Generation techniques and conversations',
             12: 'Serving and execution performance', 15: 'Reference architecture and runtime services',
             24: 'GPU execution and advanced capabilities', 31: 'Inferlets in practice'}
    index = []
    for c in chapters:
        number = c['number']
        if number in parts:
            if index:
                index.append('</ol>')
            index.append(f'<h3>{html.escape(parts[number])}</h3>')
            index.append(f'<ol class="chapter-list" start="{number}">')
        index.append(
            f'<li><a class="chapter-link" href="chapters/{c["slug"]}.html">'
            f'{html.escape(c["title"])}</a>. {html.escape(c["description"])}</li>')
    index.append('</ol>')
    intro = (ROOT / 'templates/introduction.md').read_text()
    if intro.count('{{chapter_index}}') != 1:
        raise ValueError('Introduction template must contain one chapter_index placeholder')
    intro = intro.replace('{{chapter_index}}', '\n'.join(index))
    (DOCS / 'introduction.md').write_text(intro)
    summary = ['# Summary', '', '[Introduction](introduction.md)', '']
    for c in chapters:
        if c['number'] in parts:
            summary += ['# ' + parts[c['number']], '']
        summary.append(f"- [{c['title']}](chapters/{c['slug']}.md)")
        repository = c.get('repository', REPO)
        if 'source_dir' in c:
            names = sorted(c['source_hashes'])
            blobs = [(p, (ROOT / c['source_dir'] / p).read_bytes()) for p in names]
            for path, data in blobs:
                if hashlib.sha256(data).hexdigest() != c['source_hashes'][path]:
                    raise ValueError(f'Source snapshot checksum mismatch: {path}')
        else:
            names = git('ls-tree', '-r', '--name-only', c['commit']).decode().splitlines()
            names = [p for p in names if p != 'README.md' and not p.endswith('Cargo.lock')]
            blobs = read_blobs(c['commit'], names)
        files = {}
        for path, data in blobs:
            try:
                content = data.decode('utf-8')
            except UnicodeDecodeError:
                continue
            if '\x00' in content:
                continue
            files[path] = content
            dest = DOCS / 'assets/source' / c['slug'] / (path + '.txt')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        for alias, target in c['aliases'].items():
            if target not in files:
                raise ValueError(f"Invalid alias {c['slug']}: {alias} -> {target}")
        text = ((ROOT / c['template']).read_text() if 'template' in c
                else git('show', c['commit'] + ':README.md').decode())
        text = re.sub(r'^# Chapter #\d+:', '#', text, count=1)
        text, references = link_sources(text, c, files)
        if 'comparisons' in c:
            text = render_comparisons(text, c, files, references)
        metadata = (f"\n> Source: [{c['branch']}]({repository}/tree/{c['commit']}) · "
                    f"Commit: `{c['commit'][:12]}`\n\n"
                    f"```bash\ngit checkout {c['commit']}\n```\n\n"
                    f'<span class="source-catalog" data-catalog="../assets/catalog/{c["slug"]}.json"></span>\n')
        heading, rest = text.split('\n', 1)
        (DOCS / 'chapters' / (c['slug'] + '.md')).write_text(heading + '\n' + metadata + rest)
        catalog = dict(chapter=c['title'], commit=c['commit'], slug=c['slug'],
                       repository=repository, files=sorted(files), references=references)
        (DOCS / 'assets/catalog' / (c['slug'] + '.json')).write_text(json.dumps(catalog, indent=2) + '\n')
    (DOCS / 'SUMMARY.md').write_text('\n'.join(summary) + '\n')
    print(f'Imported {len(chapters)} chapters and introduction')


if __name__ == '__main__':
    main()
