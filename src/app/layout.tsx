import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
    title: {
        template: '%s · 数据挖掘方法与实践',
        default: '数据挖掘方法与实践',
    },
    description: '大学数据挖掘课程教材——理论、算法与实践',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="zh-CN">
            <body className="bg-white text-slate-900 antialiased" suppressHydrationWarning>{children}</body>
        </html>
    )
}
