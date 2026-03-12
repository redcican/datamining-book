import { readFileSync, existsSync, statSync } from 'fs'
import { join } from 'path'
import matter from 'gray-matter'
import { renderMarkdown } from './markdown'
import { getNeighbours } from './toc'
import { slugToKey } from './toc-utils'
import type { PageData, Frontmatter } from './types'

const CONTENT_DIR = join(process.cwd(), 'content')

/**
 * Resolve a slug array to the actual file path on disk.
 * Tries several candidates in order (index.md, README.md, direct .md, etc.).
 */
function resolveFile(slug: string[]): string | null {
    if (slug.length === 0) {
        // root → preface
        const p = join(CONTENT_DIR, 'preface.md')
        return existsSync(p) ? p : null
    }
    const candidates = [
        join(CONTENT_DIR, ...slug),               // exact path (already has extension?)
        join(CONTENT_DIR, ...slug) + '.md',        // add .md extension
        join(CONTENT_DIR, ...slug, 'index.md'),    // directory index
        join(CONTENT_DIR, ...slug, 'README.md'),   // GitBook-style
    ]
    const isFile = (p: string) => existsSync(p) && statSync(p).isFile()
    return candidates.find(isFile) ?? null
}

/**
 * Load, parse, and render the page at the given slug.
 * Returns null when no file is found (→ 404).
 */
export async function getPageData(slug: string[]): Promise<PageData | null> {
    const filePath = resolveFile(slug)
    if (!filePath) return null

    const raw = readFileSync(filePath, 'utf-8')
    const { data, content: markdownBody } = matter(raw)
    const frontmatter = data as Frontmatter

    const { html, headings } = await renderMarkdown(markdownBody)
    const { prev, next } = getNeighbours(slugToKey(slug))

    return {
        slug,
        frontmatter,
        contentHtml: html,
        headings,
        prev,
        next,
    }
}
