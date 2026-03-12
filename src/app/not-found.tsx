import Link from 'next/link'

export default function NotFound() {
    return (
        <div className="min-h-screen flex flex-col items-center justify-center gap-4 text-slate-700">
            <h1 className="text-5xl font-bold text-slate-300">404</h1>
            <p className="text-lg">页面未找到 — 该章节可能尚未创建。</p>
            <Link href="/preface" className="text-blue-500 underline text-sm">
                返回前言
            </Link>
        </div>
    )
}
