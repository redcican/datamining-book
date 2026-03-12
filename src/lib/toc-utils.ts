/**
 * Pure, client-safe TOC utility functions.
 * No Node.js APIs — safe to import in Client Components.
 */

/**
 * Convert a SUMMARY.md path like "content/ch01/1.1.md" → URL slug "ch01/1.1".
 */
export function tocPathToSlug(tocPath: string): string {
    return tocPath
        .replace(/^content\//, '')
        .replace(/\/index\.md$/, '')
        .replace(/\.md$/, '')
}

/**
 * Convert a slug array ["ch01","1.1"] → string key "ch01/1.1".
 */
export function slugToKey(slug: string[]): string {
    return slug.join('/')
}
