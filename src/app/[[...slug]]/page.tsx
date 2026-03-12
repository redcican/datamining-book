import { notFound, redirect } from 'next/navigation'
import type { Metadata } from 'next'
import { getPageData } from '@/lib/content'
import { parseSummary } from '@/lib/toc'
import BookLayout from '@/components/BookLayout'

// Next.js 15: params is a Promise
type Props = { params: Promise<{ slug?: string[] }> }

export async function generateMetadata({ params }: Props): Promise<Metadata> {
    const { slug = [] } = await params
    if (slug.length === 0) return { title: '数据挖掘方法与实践' }
    const page = await getPageData(slug)
    if (!page) return { title: '页面未找到' }
    return {
        title: page.frontmatter.title ?? slug.at(-1) ?? '首页',
        description: page.frontmatter.description,
    }
}

export default async function Page({ params }: Props) {
    const { slug = [] } = await params

    // Root "/" → redirect to the preface
    if (slug.length === 0) redirect('/preface')

    const [page, toc] = await Promise.all([
        getPageData(slug),
        Promise.resolve(parseSummary()),
    ])
    if (!page) notFound()
    return <BookLayout toc={toc} page={page} />
}
