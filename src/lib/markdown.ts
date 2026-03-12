import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkRehype from 'remark-rehype'
import rehypeRaw from 'rehype-raw'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import rehypeStringify from 'rehype-stringify'
import { visit } from 'unist-util-visit'
import type { HeadingEntry } from './types'

// ---------------------------------------------------------------------------
// Block system configuration
// ---------------------------------------------------------------------------

/**
 * Tier B — Annotation hint blocks ({% hint style="X" %} ... {% endhint %})
 * All visual styling is handled by CSS classes; this set validates style names.
 */
const HINT_STYLES = new Set([
    // Backward-compatible styles (existing content uses these)
    'info', 'success', 'warning', 'danger',
    // Semantic annotation styles
    'caution',      // ⚠ 注意：常见错误、细节陷阱（warning 的语义别名）
    'tip',          // 💡 技巧：实践建议、工程提示
    'supplement',   // 🔬 补充：选读内容、深度拓展
    'proof',        // ∎  证明：数学证明，可折叠 <details> 元素
    'frontier',     // 🚀 前沿视野：2022–2026 研究进展（每章 1 个）
    'ideology',     // 🌏 课程思政：科技伦理、社会责任（每章 1 个）
    'engineering',  // 🔧 工程实践：工业界经验与最佳实践（每章 1 个）
    'mathbg',       // 📐 数学基础：就地引入必要数学工具
    'think',        // 🤔 思考：引导性问题，激活主动阅读
])

/**
 * Tier A — Formal numbered blocks ({% block type="X" label="N" title="T" %})
 * Maps English type name → Chinese badge label.
 */
const BLOCK_BADGE: Record<string, string> = {
    definition: '定义',
    theorem:    '定理',
    corollary:  '推论',
    lemma:      '引理',
    example:    '例',
    algorithm:  '算法',
    case:       '案例',
}

// ---------------------------------------------------------------------------
// Render a markdown fragment through the full pipeline (no block processing).
// Used to render block bodies so math and GFM work inside all block types.
// ---------------------------------------------------------------------------
async function renderFragment(md: string): Promise<string> {
    const processor = unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkMath)
        .use(remarkRehype, { allowDangerousHtml: true })
        .use(rehypeRaw)
        .use(rehypeKatex)
        .use(rehypeStringify)
    const result = await processor.process(md)
    return String(result)
}

// ---------------------------------------------------------------------------
// Parse key="value" attribute pairs from a {% block %} attribute string.
// ---------------------------------------------------------------------------
function parseAttrs(s: string): Record<string, string> {
    const attrs: Record<string, string> = {}
    const re = /(\w+)="([^"]*)"/g
    let m: RegExpExecArray | null
    while ((m = re.exec(s)) !== null) attrs[m[1]] = m[2]
    return attrs
}

// ---------------------------------------------------------------------------
// Pre-processing: transform custom block syntax → HTML.
// Processes Tier A ({% block %}) then Tier B ({% hint %}).
// Block bodies are rendered through renderFragment() so that math ($$),
// GFM tables, and bold/italic all work inside any block type.
// ---------------------------------------------------------------------------
async function preprocessGitBook(md: string): Promise<string> {

    // ── Tier A: Formal numbered blocks ──────────────────────────────────────
    //  {% block type="definition" label="1.1" title="知识发现（KDD）" %}
    //  ...body (supports math, GFM, code blocks)...
    //  {% endblock %}
    const blockRe = /\{%\s*block\s+([^%]+?)\s*%\}([\s\S]*?)\{%\s*endblock\s*%\}/g
    const formalBlocks: Array<{ original: string; attrs: Record<string, string>; body: string }> = []
    for (const m of md.matchAll(blockRe)) {
        formalBlocks.push({ original: m[0], attrs: parseAttrs(m[1]), body: m[2] })
    }
    for (const { original, attrs, body } of formalBlocks) {
        const type      = attrs.type  ?? 'definition'
        const label     = attrs.label ?? ''
        const title     = attrs.title ?? ''
        const badge     = BLOCK_BADGE[type] ?? type
        const badgeText = label ? `${badge}\u00a0${label}` : badge   // non-breaking space
        const bodyHtml  = await renderFragment(body.trim())
        const titleHtml = title ? `<span class="dm-block-title">${title}</span>` : ''
        const html =
            `<div class="dm-block dm-block-${type}">` +
            `<div class="dm-block-header">` +
            `<span class="dm-block-badge">${badgeText}</span>${titleHtml}` +
            `</div>` +
            `<div class="dm-block-body">${bodyHtml}</div>` +
            `</div>`
        md = md.replace(original, () => html)
    }

    // ── Tier B: Annotation hint blocks ──────────────────────────────────────
    //  {% hint style="caution" %} ... {% endhint %}
    //  'proof' style renders as a collapsible <details> element.
    const hintRe = /\{%\s*hint\s+style="(\w+)"\s*%\}([\s\S]*?)\{%\s*endhint\s*%\}/g
    const hints: Array<{ original: string; style: string; body: string }> = []
    for (const m of md.matchAll(hintRe)) {
        hints.push({ original: m[0], style: m[1], body: m[2] })
    }
    for (const { original, style, body } of hints) {
        const cls      = HINT_STYLES.has(style) ? style : 'info'
        const bodyHtml = await renderFragment(body.trim())
        let html: string
        if (cls === 'proof') {
            // Collapsible proof block — renders as <details>
            html =
                `<details class="dm-hint dm-hint-proof">` +
                `<summary class="dm-proof-summary">` +
                `<span class="dm-proof-icon">∎</span>证明` +
                `</summary>` +
                `<div class="dm-proof-body">${bodyHtml}</div>` +
                `</details>`
        } else {
            html = `<div class="dm-hint dm-hint-${cls}">${bodyHtml}</div>`
        }
        md = md.replace(original, () => html)
    }

    // ── Tabs ─────────────────────────────────────────────────────────────────
    md = md.replace(/\{%\s*tabs\s*%\}/g, '<div class="dm-tabs">')
    md = md.replace(/\{%\s*endtabs\s*%\}/g, '</div>')
    md = md.replace(/\{%\s*tab\s+title="([^"]+)"\s*%\}/g, '<div class="dm-tab" data-title="$1">')
    md = md.replace(/\{%\s*endtab\s*%\}/g, '</div>')

    // ── Bare GitBook code fences ──────────────────────────────────────────────
    md = md.replace(/\{%\s*code[^%]*%\}/g, '')
    md = md.replace(/\{%\s*endcode\s*%\}/g, '')

    // ── Figure captions: render math inside <figcaption> ────────────────────
    // remark-math does not scan raw HTML nodes, so $...$ inside <figcaption>
    // is never converted to math AST nodes and rehype-katex skips it.
    // Solution: pre-render figcaption bodies through renderFragment().
    const figcaptionRe = /<figcaption>([\s\S]*?)<\/figcaption>/g
    const figcaptions: Array<{ original: string; body: string }> = []
    for (const m of md.matchAll(figcaptionRe)) {
        figcaptions.push({ original: m[0], body: m[1] })
    }
    for (const { original, body } of figcaptions) {
        const bodyHtml = await renderFragment(body.trim())
        // renderFragment wraps output in <p>; unwrap to keep figcaption inline
        const inner = bodyHtml.replace(/^\s*<p>([\s\S]*?)<\/p>\s*$/, '$1')
        md = md.replace(original, `<figcaption>${inner}</figcaption>`)
    }

    // ── Figure placeholders ───────────────────────────────────────────────────
    md = md.replace(
        /<!--\s*FIGURE PLACEHOLDER\s*-->([\s\S]*?)<!--\s*END PLACEHOLDER\s*-->/g,
        (_, inner) => `<div class="dm-figure-placeholder">${inner.trim()}</div>`
    )

    return md
}

// ---------------------------------------------------------------------------
// Heading slug generator (works for CJK + ASCII)
// ---------------------------------------------------------------------------
function headingId(text: string): string {
    return text
        .toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/[^\w\u4e00-\u9fff-]/g, '')
        .replace(/^-+|-+$/g, '')
}

function extractText(node: any): string {
    if (node.type === 'text' || node.type === 'raw') return node.value ?? ''
    if (node.children) return (node.children as any[]).map(extractText).join('')
    return ''
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------
export async function renderMarkdown(
    rawMarkdown: string
): Promise<{ html: string; headings: HeadingEntry[] }> {
    const preprocessed = await preprocessGitBook(rawMarkdown)
    const headings: HeadingEntry[] = []

    const processor = unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkMath)
        .use(remarkRehype, { allowDangerousHtml: true })
        .use(rehypeRaw)
        .use(rehypeKatex)
        .use(rehypeHighlight, { detect: true })
        // Wrap <pre><code class="language-X"> in <div class="dm-code-block" data-lang="X">
        .use(() => (tree) => {
            visit(tree, 'element', (node: any, index: any, parent: any) => {
                if (node.tagName !== 'pre' || index == null || !parent) return
                const code = node.children?.[0]
                if (code?.tagName !== 'code') return
                const langClass = (code.properties?.className as string[] ?? [])
                    .find((c: string) => c.startsWith('language-'))
                const lang = langClass?.replace('language-', '') ?? ''
                if (!lang) return
                parent.children[index] = {
                    type: 'element',
                    tagName: 'div',
                    properties: { className: ['dm-code-block'], 'data-lang': lang },
                    children: [node],
                }
            })
        })
        // Inject IDs into headings and collect TOC entries
        .use(() => (tree) => {
            visit(tree, 'element', (node: any) => {
                if (/^h[1-4]$/.test(node.tagName)) {
                    const level = parseInt(node.tagName[1])
                    const text  = extractText(node)
                    const id    = headingId(text)
                    node.properties = { ...node.properties, id }
                    headings.push({ id, text, level })
                }
            })
        })
        .use(rehypeStringify)

    const result = await processor.process(preprocessed)
    return { html: String(result), headings }
}
