'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { useState } from 'react'
import type { TocEntry } from '@/lib/types'
import { tocPathToSlug } from '@/lib/toc-utils'

// ── Chapter group (collapsible) ───────────────────────────────────────────
function ChapterGroup({ entry, depth = 0 }: { entry: TocEntry; depth?: number }) {
    const pathname = usePathname()
    const slug = '/' + tocPathToSlug(entry.path)
    const isActive = pathname === slug
    const hasChildren = entry.children.length > 0

    // A chapter is "open" if the current page is anywhere inside it
    const isOpen = hasChildren && entry.children.some((c) => isDescendant(c, pathname))
    const [open, setOpen] = useState(isOpen)

    if (!hasChildren) {
        // Leaf node
        return (
            <Link
                href={slug}
                className={[
                    'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors duration-100',
                    depth === 0 ? 'font-medium' : 'font-normal opacity-90',
                    isActive
                        ? 'bg-blue-500/15 text-blue-400'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/5',
                ].join(' ')}
                style={{ paddingLeft: `${12 + depth * 14}px` }}
            >
                {isActive && <span className="w-1 h-1 rounded-full bg-blue-400 shrink-0" />}
                {entry.title}
            </Link>
        )
    }

    return (
        <div>
            <button
                onClick={() => setOpen((o) => !o)}
                className={[
                    'w-full flex items-center justify-between px-3 py-1.5 rounded-md text-sm font-semibold transition-colors',
                    'text-slate-300 hover:text-slate-100 hover:bg-white/5',
                ].join(' ')}
                style={{ paddingLeft: `${12 + depth * 14}px` }}
            >
                <span>{entry.title}</span>
                <svg
                    className={`w-3.5 h-3.5 text-slate-500 transition-transform ${open ? 'rotate-90' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
                >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
            </button>
            {open && (
                <div className="mt-0.5">
                    {entry.children.map((child) => (
                        <ChapterGroup key={child.path} entry={child} depth={depth + 1} />
                    ))}
                </div>
            )}
        </div>
    )
}

function isDescendant(entry: TocEntry, pathname: string): boolean {
    const slug = '/' + tocPathToSlug(entry.path)
    if (slug === pathname) return true
    return entry.children.some((c) => isDescendant(c, pathname))
}

// ── Main Sidebar ──────────────────────────────────────────────────────────
export default function Sidebar({ toc }: { toc: TocEntry[] }) {
    return (
        <nav
            aria-label="目录"
            className="sidebar fixed top-0 left-0 h-screen flex flex-col overflow-hidden"
            style={{ background: 'var(--color-sidebar-bg)', borderRight: '1px solid #1e293b' }}
        >
            {/* Book title */}
            <div className="px-5 py-5 border-b border-slate-800 shrink-0">
                <Link href="/preface" className="block">
                    <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 mb-1">
                        教材
                    </p>
                    <h1 className="text-sm font-bold text-slate-200 leading-snug">
                        数据挖掘方法与实践
                    </h1>
                </Link>
            </div>

            {/* Navigation tree */}
            <div className="flex-1 overflow-y-auto py-4 px-2 space-y-0.5">
                {toc.map((entry) => (
                    <ChapterGroup key={entry.path} entry={entry} />
                ))}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-slate-800 shrink-0">
                <p className="text-[10px] text-slate-600">© 2025 教材编写组</p>
            </div>
        </nav>
    )
}
