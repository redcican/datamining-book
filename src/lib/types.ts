export interface TocEntry {
    title: string
    path: string // relative path as written in SUMMARY.md, e.g. "content/ch01/index.md"
    level: number
    children: TocEntry[]
}

export interface Frontmatter {
    title?: string
    description?: string
    status?: 'draft' | 'in-progress' | 'done'
    section?: string
}

export interface HeadingEntry {
    id: string
    text: string
    level: number
}

export interface NavItem {
    title: string
    path: string // slug array joined, e.g. "ch01/1.1"
}

export interface PageData {
    slug: string[]
    frontmatter: Frontmatter
    contentHtml: string
    headings: HeadingEntry[]
    prev?: NavItem
    next?: NavItem
}
