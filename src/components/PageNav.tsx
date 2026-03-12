import Link from 'next/link'
import type { NavItem } from '@/lib/types'

export default function PageNav({ prev, next }: { prev?: NavItem; next?: NavItem }) {
    if (!prev && !next) return null
    return (
        <nav className="mt-14 pt-6 border-t border-slate-200 flex justify-between gap-4">
            {prev ? (
                <Link
                    href={`/${prev.path}`}
                    className="group flex flex-col items-start max-w-xs px-4 py-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
                >
                    <span className="text-[11px] text-slate-400 mb-0.5">← 上一节</span>
                    <span className="text-sm font-medium text-slate-700 group-hover:text-blue-600">
                        {prev.title}
                    </span>
                </Link>
            ) : (
                <div />
            )}

            {next ? (
                <Link
                    href={`/${next.path}`}
                    className="group flex flex-col items-end max-w-xs px-4 py-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors ml-auto"
                >
                    <span className="text-[11px] text-slate-400 mb-0.5">下一节 →</span>
                    <span className="text-sm font-medium text-slate-700 group-hover:text-blue-600">
                        {next.title}
                    </span>
                </Link>
            ) : null}
        </nav>
    )
}
