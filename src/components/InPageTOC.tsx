'use client'

import { useEffect, useState } from 'react'
import type { HeadingEntry } from '@/lib/types'

export default function InPageTOC({ headings }: { headings: HeadingEntry[] }) {
    const [active, setActive] = useState<string>('')

    useEffect(() => {
        if (headings.length === 0) return
        const observer = new IntersectionObserver(
            (entries) => {
                for (const e of entries) {
                    if (e.isIntersecting) {
                        setActive(e.target.id)
                        break
                    }
                }
            },
            { rootMargin: '-20px 0% -70% 0%' }
        )
        headings.forEach(({ id }) => {
            const el = document.getElementById(id)
            if (el) observer.observe(el)
        })
        return () => observer.disconnect()
    }, [headings])

    if (headings.length < 2) return null

    // Only show h2 and h3
    const visible = headings.filter((h) => h.level <= 3)

    return (
        <aside
            className="toc-panel hidden xl:block fixed top-0 right-0 h-screen overflow-y-auto py-8 px-4"
            style={{ width: 'var(--toc-w)' }}
        >
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400 mb-3">
                本页目录
            </p>
            <nav className="in-page-toc space-y-0.5">
                {visible.map((h) => (
                    <a
                        key={h.id}
                        href={`#${h.id}`}
                        className={`block text-sm py-0.5 transition-colors ${
                            h.level === 3 ? 'pl-3' : ''
                        } ${active === h.id ? 'active text-blue-500 font-medium' : 'text-slate-500 hover:text-slate-800'}`}
                    >
                        {h.text}
                    </a>
                ))}
            </nav>
        </aside>
    )
}
