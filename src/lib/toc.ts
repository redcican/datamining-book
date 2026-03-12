import { readFileSync } from 'fs'
import { join } from 'path'
import type { TocEntry, NavItem } from './types'
import { tocPathToSlug } from './toc-utils'

// Re-export for convenience
export { tocPathToSlug, slugToKey } from './toc-utils'

const SUMMARY_PATH = join(process.cwd(), 'SUMMARY.md')

/**
 * Parse SUMMARY.md into a nested TocEntry tree.
 * Supports two link syntaxes:
 *   - [Title](content/ch01/index.md)
 *   - [Title](ch01/index.md)   ← also accepted, normalised to no leading "content/"
 */
export function parseSummary(): TocEntry[] {
    const raw = readFileSync(SUMMARY_PATH, 'utf-8')
    const lines = raw.split('\n')
    const roots: TocEntry[] = []
    const stack: TocEntry[] = []

    for (const line of lines) {
        const match = line.match(/^(\s*)-\s+\[([^\]]+)\]\(([^)]+)\)/)
        if (!match) continue
        const [, indent, title, rawPath] = match
        const path = rawPath.trim()
        const level = Math.floor(indent.length / 2)
        const entry: TocEntry = { title, path, level, children: [] }

        if (level === 0) {
            roots.push(entry)
            stack.length = 0
            stack.push(entry)
        } else {
            // pop stack back to the correct parent level
            while (stack.length > level) stack.pop()
            const parent = stack[stack.length - 1]
            if (parent) parent.children.push(entry)
            stack.push(entry)
        }
    }
    return roots
}

/** Flatten the tree to a depth-first ordered list (only leaf-ish entries with real paths) */
export function flatToc(entries: TocEntry[]): TocEntry[] {
    const out: TocEntry[] = []
    function walk(e: TocEntry) {
        out.push(e)
        e.children.forEach(walk)
    }
    entries.forEach(walk)
    return out
}

/**
 * Given the current page's slug (joined as "ch01/1.1"), return the
 * previous and next navigable pages.
 */
export function getNeighbours(currentSlug: string): { prev?: NavItem; next?: NavItem } {
    const all = flatToc(parseSummary()).filter((e) => e.path.endsWith('.md'))
    const idx = all.findIndex((e) => tocPathToSlug(e.path) === currentSlug)
    if (idx < 0) return {}
    const toNav = (e: TocEntry): NavItem => ({ title: e.title, path: tocPathToSlug(e.path) })
    return {
        prev: idx > 0 ? toNav(all[idx - 1]) : undefined,
        next: idx < all.length - 1 ? toNav(all[idx + 1]) : undefined,
    }
}

